import numpy as np
import cv2
import os
import configparser
import struct

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
    interpolation=cv2.INTER_AREA
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

    # 패키지 외 임시파일 삭제 (필요없다면 이 블록 주석처리)
    for p in pkg_files:
        os.remove(p)

    return pkg_path  # 패키지 파일 경로 반환
