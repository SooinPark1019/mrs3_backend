from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
import os
import uuid
from pathlib import Path
import shutil
import cv2

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
    pkgimg.restore_img_mult_tgs_server(
        input_path=unpacked_dir,
        mrs3_mode=mrs3_mode,
        output_path=TEMP_DIR
    )
    restored_image_path = os.path.join(TEMP_DIR, "restored.png")
    if not os.path.exists(restored_image_path):
        raise HTTPException(status_code=500, detail="복원 이미지 생성 실패")

    # [4] 복원 이미지 반환
    return FileResponse(restored_image_path, filename="restored.png", media_type="image/png")

@app.post("/auto-batch-compress")
async def auto_batch_compress(
    images: list[UploadFile] = File(...),
    scaler: int = Form(2),
    manual: bool = Form(True),
    background_tasks: BackgroundTasks = None
):
    """
    여러 이미지를 받아 임시 폴더에 저장 → compress_mult_img_server로 처리 → zip 반환
    manual: True(수동, ROI별도입력), False(자동: YOLO 등)
    """
    ensure_temp_dir()
    session_id = str(uuid.uuid4())
    temp_dir = os.path.join(TEMP_DIR, f"session_{session_id}")
    os.makedirs(temp_dir, exist_ok=True)
    output_zip_path = os.path.join(TEMP_DIR, f"pkgs_{session_id}.zip")

    for image in images:
        unique_name = f"{uuid.uuid4().hex}_{Path(image.filename).name}"
        image_path = os.path.join(temp_dir, unique_name)
        with open(image_path, "wb") as f:
            f.write(await image.read())

    interpolation = cv2.INTER_AREA

    imgpkg.compress_mult_img_server(
        input_path=temp_dir,
        output_path=output_zip_path,
        manual=manual,
        scaler=scaler,
        interpolation=interpolation
    )

    if not os.path.exists(output_zip_path):
        raise HTTPException(status_code=500, detail="압축 zip 생성 실패")

    background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)
    background_tasks.add_task(os.remove, output_zip_path)

    return FileResponse(output_zip_path, filename="pkgs.zip", media_type="application/zip")

@app.post("/batch-restore")
async def batch_restore(
    pkgs_zip: UploadFile = File(...),
    mrs3_mode: int = Form(-1),
    background_tasks: BackgroundTasks = None
):
    """
    여러 개의 .pkg 파일이 들어있는 zip 파일을 업로드 받아
    모든 패키지를 복원(png)하여 zip으로 반환하는 API.
    - pkgs_zip: zip 파일(.pkg들이 들어있음)
    - mrs3_mode: 업스케일 복원 옵션
    """
    ensure_temp_dir()
    session_id = str(uuid.uuid4())
    temp_dir = os.path.join(TEMP_DIR, f"restore_session_{session_id}")
    os.makedirs(temp_dir, exist_ok=True)

    # 1. zip 저장 및 압축 해제
    zip_path = os.path.join(temp_dir, "input_pkgs.zip")
    with open(zip_path, "wb") as f:
        f.write(await pkgs_zip.read())
    pkgs_dir = os.path.join(temp_dir, "pkgs_unzipped")
    os.makedirs(pkgs_dir, exist_ok=True)
    shutil.unpack_archive(zip_path, pkgs_dir)  # zip 해제

    # 2. 복원 이미지 저장 폴더 준비
    restored_dir = os.path.join(temp_dir, "restored_pngs")
    os.makedirs(restored_dir, exist_ok=True)

    # 3. pkgimg.restore_imgs_in_folder_server 호출
    # (input_path: pkg 폴더, output_path: 복원폴더, mrs3_mode)
    pkgimg.restore_imgs_in_folder_server(
        input_path=pkgs_dir,
        output_path=restored_dir,
        mrs3_mode=mrs3_mode
    )

    # 4. 복원된 png들을 zip으로 묶기
    restored_zip_path = os.path.join(temp_dir, "restored_imgs.zip")
    shutil.make_archive(restored_zip_path[:-4], 'zip', restored_dir)

    # 5. 반환
    if not os.path.exists(restored_zip_path):
        raise HTTPException(status_code=500, detail="복원 zip 생성 실패")

    background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)
    background_tasks.add_task(os.remove, restored_zip_path)

    return FileResponse(restored_zip_path, filename="restored_imgs.zip", media_type="application/zip")