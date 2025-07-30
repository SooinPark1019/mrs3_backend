from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from pathlib import Path

from validators import parse_polygons, validate_scaler

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'MRS3/ServerInterface'))
import MRS3.ServerInterface.ImgToPkg_Interface as imgpkg
import MRS3.ServerInterface.PkgToImg_Interface as pkgimg

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://www.photoshrink.shop", "https://photoshrink.shop"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp"

def ensure_temp_dir():
    """임시 폴더가 없으면 생성."""
    os.makedirs(TEMP_DIR, exist_ok=True)

def get_unique_path(filename: str, suffix: str = "") -> str:
    """uuid와 원본 파일명, 옵션 suffix로 유니크 경로 생성."""
    session_id = str(uuid.uuid4())
    safe_name = Path(filename).name  # 보안: 디렉토리 오염 방지
    return os.path.join(TEMP_DIR, f"{session_id}_{suffix}{safe_name}")

async def save_upload_file(upload_file: UploadFile, dest_path: str):
    """업로드 파일을 지정 경로에 저장 (비동기 지원)."""
    with open(dest_path, "wb") as f:
        f.write(await upload_file.read())

@app.post("/compress")
async def compress_image(
    image: UploadFile = File(...),
    polygons: str = Form(...),
    scaler: int = Form(2),
    use_imgpresso: bool = Form(False),
):
    """
    업로드된 이미지와 ROI(관심영역) 정보로 패키지(.pkg) 파일을 생성하여 반환합니다.

    - 이미지에서 지정된 다각형 영역들(ROI)만 추출하여 압축 및 패키징합니다.
    - `scaler` 값으로 전체 이미지를 downscaling 할 수 있습니다.
    - `use_imgpresso=True`로 고효율 압축(imgpresso)을 적용할 수 있습니다(선택).
    - 반환값은 .pkg 확장자의 바이너리 패키지 파일입니다.

    파라미터
    - image: 업로드할 원본 이미지 파일 (필수)
    - polygons: ROI 좌표 리스트 (JSON 문자열, 예: [[[x1, y1], [x2, y2], ...], ...])
    - scaler: 다운스케일 배율 (2, 3, 4 중 선택, 기본값 2)
    - use_imgpresso: 고효율 압축(imgpresso) 적용 여부 (기본값: False)

    반환
    - output.pkg: 압축된 결과물 패키지 파일 (application/octet-stream)
    """
    polygons_data = parse_polygons(polygons)
    scaler = validate_scaler(scaler)

    ensure_temp_dir()
    image_path = get_unique_path(image.filename)
    await save_upload_file(image, image_path)

    kwargs = dict(
        img_path=image_path,
        output_path=TEMP_DIR,
        scaler=scaler,
        roi_point_lists=polygons_data,
        pkg_filename="output.pkg"
    )

    if use_imgpresso:
        pkg_path = imgpkg.compress_img_pkg_imgpresso(**kwargs)
    else:
        pkg_path = imgpkg.compress_img_mult_tgs_server(**kwargs)

    if not pkg_path or not os.path.exists(pkg_path):
        raise HTTPException(status_code=500, detail="패키지 생성 실패")

    return FileResponse(pkg_path, filename="output.pkg", media_type="application/octet-stream")



@app.post("/restore")
async def restore_image(
    pkg: UploadFile = File(...),
    mrs3_mode: int = Form(-1)  # EDSR: -1, 그 외는 cv2 인터폴레이션 번호
):
    """
    .pkg 파일을 받아 복원 이미지를 반환하는 API

    - pkg: 업로드된 .pkg 파일
    - mrs3_mode: 업스케일 방식(-1=EDSR, 0/1/2 등은 cv2 방식)
    """
    # [1] 임시 파일 경로 생성 및 저장
    ensure_temp_dir()
    pkg_path = get_unique_path(pkg.filename)
    await save_upload_file(pkg, pkg_path)

    # [2] 언팩 디렉토리 생성(유니크)
    unpacked_dir = get_unique_path(pkg.filename, suffix="unpacked_")
    os.makedirs(unpacked_dir, exist_ok=True)

    # [3] .pkg 언팩 및 복원 이미지 생성
    pkgimg.unpack_files(pkg_path, unpacked_dir)
    pkgimg.restore_img_mult_tgs(
        input_path=unpacked_dir,
        mrs3_mode=mrs3_mode,
        output_path=TEMP_DIR
    )
    restored_image_path = os.path.join(TEMP_DIR, "restored.png")
    if not os.path.exists(restored_image_path):
        raise HTTPException(status_code=500, detail="복원 이미지 생성 실패")

    # [4] 복원 이미지 반환
    return FileResponse(restored_image_path, filename="restored.png", media_type="image/png")
