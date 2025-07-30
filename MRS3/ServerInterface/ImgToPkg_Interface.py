import numpy as np
import cv2
import os
import configparser
import struct
import Utils
from ultralytics import YOLO

# 파일명 상수
roi_filename = 'roi'
roi_binary_filename = 'bin'
downscaled_filename = 'downscaled'
config_filename = 'config'

def pack_files_server(output_file: str, input_files: list):
    """
    여러 파일을 하나의 사용자 정의 패키지(.pkg)로 묶습니다.

    Args:
        output_file (str): 출력 패키지 파일 경로 (예: 'output.pkg')
        input_files (list): 패키징할 파일 경로들의 리스트
    """
    with open(output_file, 'wb') as f_out:
        # 파일 개수 기록 (4바이트)
        f_out.write(struct.pack('I', len(input_files)))
        for file_path in input_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일을 찾을 수 없음: {file_path}")
            file_name = os.path.basename(file_path)
            with open(file_path, 'rb') as f_in:
                file_data = f_in.read()
            encoded_name = file_name.encode('utf-8')
            # 파일명 길이 및 파일명 기록
            f_out.write(struct.pack('I', len(encoded_name)))
            f_out.write(encoded_name)
            # 파일 데이터 크기 및 파일 데이터 기록
            f_out.write(struct.pack('I', len(file_data)))
            f_out.write(file_data)

def _select_multiple_polygon_roi_server(image_path, roi_point_lists):
    """
    다각형 꼭짓점 좌표 리스트로부터 각 ROI 영역(이미지, 마스크, 좌표범위)을 추출합니다.

    Args:
        image_path (str): 원본 이미지 경로
        roi_point_lists (list): 각 ROI의 꼭짓점 좌표 리스트들의 리스트
            예: [
                    [[x1, y1], [x2, y2], [x3, y3], ...],   # 첫 번째 ROI
                    [[x1, y1], [x2, y2], [x3, y3], ...],   # 두 번째 ROI
                    ...
                ]

    Returns:
        list: [
                (cropped_img, cropped_mask, (y_from, y_to, x_from, x_to)),
                ...
            ]
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지 로딩 실패: {image_path}")

    result = []
    for points in roi_point_lists:
        if len(points) < 3:
            print("3개 이상의 꼭짓점이 필요합니다.")
            continue
        # 좌표를 int32로 변환
        points_np = np.array(points, dtype=np.int32)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points_np], 255)
        roi = cv2.bitwise_and(img, img, mask=mask)
        x, y, w, h = cv2.boundingRect(points_np)
        cropped = roi[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]
        result.append((cropped, cropped_mask, (y, y+h, x, x+w)))
    return result

def compress_img_mult_tgs_server(
    img_path,
    output_path,
    scaler,
    roi_point_lists,      # 프론트에서 받은 [[(x1, y1), ...], ...]
    pkg_filename='output.pkg',
    interpolation=cv2.INTER_AREA,
    delete_temp=True,
):
    """
    여러 ROI를 압축해 패키지(.pkg) 파일로 저장합니다.

    Args:
        img_path (str): 원본 이미지 경로
        output_path (str): 결과 저장 폴더 경로
        scaler (int): downscaling 배율
        roi_point_lists (list): 다각형 ROI 좌표 리스트들의 리스트
        pkg_filename (str): 패키지 파일명 (default: 'output.pkg')
        interpolation: 다운스케일에 사용할 interpolation 방식 (default: cv2.INTER_AREA)

    Returns:
        str: 생성된 패키지 파일 경로
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image: {img_path}")
        return None

    # 각 ROI에 대해 원본 부분 이미지, 마스크, 좌표 추출
    targets = _select_multiple_polygon_roi_server(img_path, roi_point_lists)

    # 다운스케일 이미지 저장
    downscaled_path = os.path.join(output_path, f'{downscaled_filename}.png')
    downscaled = cv2.resize(
        img,
        (img.shape[1] // scaler, img.shape[0] // scaler),
        interpolation=interpolation
    )
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv2.imwrite(downscaled_path, downscaled, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    # 메타데이터 ini 저장
    config = configparser.ConfigParser()
    config['DEFAULT'] = {
        'SCALER': f'{scaler}',
        'NUMBER_OF_TARGETS': f'{len(targets)}'
    }
    for i, t in enumerate(targets):
        config[f'{i}'] = {
            'Y_FROM': f'{t[2][0]}',
            'Y_TO': f'{t[2][1]}',
            'X_FROM': f'{t[2][2]}',
            'X_TO': f'{t[2][3]}'
        }
    config_path = os.path.join(output_path, f'{config_filename}.ini')
    with open(config_path, 'w') as configfile:
        config.write(configfile)

    # 각 ROI 이미지 및 마스크 저장
    roi_img_paths = []
    roi_mask_paths = []
    for i, t in enumerate(targets):
        roi_img_path = os.path.join(output_path, f'{roi_filename}{i}.png')
        roi_mask_path = os.path.join(output_path, f'{roi_binary_filename}{i}.png')
        cv2.imwrite(roi_img_path, t[0], [cv2.IMWRITE_PNG_COMPRESSION, 9])
        cv2.imwrite(roi_mask_path, t[1], [cv2.IMWRITE_PNG_COMPRESSION, 9, cv2.IMWRITE_PNG_BILEVEL, 1])
        roi_img_paths.append(roi_img_path)
        roi_mask_paths.append(roi_mask_path)

    # 패키지에 넣을 파일 목록 생성
    pkg_files = [config_path, downscaled_path] + roi_img_paths + roi_mask_paths
    pkg_path = os.path.join(output_path, pkg_filename)
    pack_files_server(pkg_path, pkg_files)

    if delete_temp:
        for p in pkg_files:
            try:
                os.remove(p)
            except Exception as e:
                print(f"[임시파일 삭제 오류] {p} - {e}")

    return pkg_path  # 패키지 파일 경로 반환

def compress_img_pkg_imgpresso(
    img_path,
    output_path,
    scaler,
    roi_point_lists,
    pkg_filename='output.pkg',
    interpolation=cv2.INTER_AREA
):
    """
    imgpresso 방식 압축
    """
    # [1] 일반 패키징(ROI 마스킹, 다운스케일, ini, 패키지파일 생성 등)
    compress_img_mult_tgs_server(
        img_path=img_path,
        output_path=output_path,
        scaler=scaler,
        roi_point_lists=roi_point_lists,
        pkg_filename=pkg_filename,
        interpolation=interpolation,
        delete_temp=False,
    )

    # [2] config.ini에서 ROI 개수 읽기
    config = configparser.ConfigParser()
    config.read(os.path.join(output_path, f"{config_filename}.ini"))
    target_num = int(config['DEFAULT']['NUMBER_OF_TARGETS'])

    # [3] imgpresso 압축(예: utils.compress_imgpresso_replace 함수)
    Utils.compress_imgpresso_replace(os.path.join(output_path, f"{downscaled_filename}.png"), output_path)
    for i in range(target_num):
        Utils.compress_imgpresso_replace(os.path.join(output_path, f"{roi_filename}{i}.png"), output_path)

    pkg_files = [os.path.join(output_path, f"{config_filename}.ini"), os.path.join(output_path, f"{downscaled_filename}.png")]
    for i in range(target_num):
        pkg_files.append(os.path.join(output_path, f"{roi_filename}{i}.png"))
        pkg_files.append(os.path.join(output_path, f"{roi_binary_filename}{i}.png"))

    Utils.pack_files(output_file=os.path.join(output_path, pkg_filename), input_files=pkg_files)

    for pfile in pkg_files:
        try:
            os.unlink(pfile)
        except Exception as e:
            print(f"[임시파일 삭제 오류] {pfile} - {e}")

    
    return os.path.join(output_path, pkg_filename)


model = YOLO('../models/yolov8m-face-lindevs.pt')


def compress_mult_img_server(input_path, output_path, manual=True, scaler=4, interpolation=mr.INTER_AREA):
    """
    :param input_path: 수동 압축 이미지 모아놓은 폴더 경로
    :param output_path: pkg 결과 저장 폴더 경로
    :param manual: 해당 폴더 처리 자동/수동 여부. True 면 수동, False 면 자동.
    :param scaler: 이미지 shrink scaler
    :param interpolation: shrink 에 적용할 interpolation manner
    """
    for filename in os.listdir(input_path):
        if filename.lower().endswith('.png'):
            full_path = os.path.join(input_path, filename)
            
            filename_with_ext = os.path.basename(filename)
            pkg_filename_split, _ = os.path.splitext(filename_with_ext)
            pkg_filename = f'{pkg_filename_split}.pkg'

            roi_point_lists = []
            if manual: # 수동 타겟
                # TODO: 현재 full_path 이미지에 대응하는 roi_point_lists 가져오기 - 아래 코드 지우고, 웹형식에 맞게 수정 필요
                # roi_point_lists = _select_multiple_polygon_roi(full_path)
                pass
            else: # 자동 타겟
                img = cv2.imread(full_path)
                results = model(img)
                
                for box in results[0].boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    roi_point_lists.append([(x1,y1), (x2,y1), (x2,y2), (x1,y2)])
            
            compress_img_mult_tgs_server(img_path=full_path, 
                                            output_path=output_path, 
                                            scaler=scaler, 
                                            pkg_filename=pkg_filename,
                                            roi_point_lists=roi_point_lists,
                                            interpolation=interpolation,
                                            delete_temp=True)
    return


