import os
import struct
import configparser
import cv2
import numpy as np
import Interpolation as inter
import time
from collections.abc import Callable
import gc

# 파일명/상수 정의
roi_filename = 'roi'
roi_binary_filename = 'bin'
downscaled_filename = 'downscaled'
config_filename = 'config'
restored_filename = 'restored'

BLEND_LINEAR = 0
BLEND_HERMIT_3 = 1
BLEND_HERMIT_5 = 2
BLEND_SINUSOIDAL = 3
BLEND_STEP = 4
_DIST_CRIT_COEF = .4
PIXELS_LIMIT = 281000
OVERLAP_HALF_LENGTH = 20

def unpack_files(input_file: str, output_dir: str):
    """
    여러 파일이 묶인 사용자 정의 패키지(.pkg) 파일에서 원본 파일을 추출합니다.

    Args:
        input_file (str): 입력 패키지 파일 경로 (예: 'output.pkg')
        output_dir (str): 파일이 추출될 디렉토리 경로 (예: 'unpacked')
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(input_file, 'rb') as f_in:
        num_files = struct.unpack('I', f_in.read(4))[0]
        for _ in range(num_files):
            name_len = struct.unpack('I', f_in.read(4))[0]
            file_name = f_in.read(name_len).decode('utf-8')
            data_len = struct.unpack('I', f_in.read(4))[0]
            file_data = f_in.read(data_len)
            output_path = os.path.join(output_dir, file_name)
            with open(output_path, 'wb') as f_out:
                f_out.write(file_data)

def _upscale_by_edsr(image_path, scaler):
    """
    EDSR 슈퍼레졸루션으로 업스케일링 (CUDA 필요, 미사용시 None 반환)
    """
    t1 = time.time()
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    if scaler not in [2, 3, 4]:
        print(f"Invalid scaler value: {scaler}. Must be 2, 3 or 4.")
        return None
    if not cv2.cuda.getCudaEnabledDeviceCount():
        print("No CUDA-enabled GPU found.")
        return None

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    try:
        sr.readModel(f'models/EDSR_x{scaler}.pb')
    except Exception as e:
        print(f"Error reading model: {e}")
        return None

    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    sr.setModel('edsr', scaler)
    try:
        result = sr.upsample(img)
    except Exception as e:
        print(f"Error during upscaling: {e}")
        return None
    t2 = time.time()
    print(f'{t2-t1} sec taken')
    return result

def _upscale_by_resize(image_path, scaler, interpolation=cv2.INTER_CUBIC):
    """
    일반 resize로 업스케일링
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    h, w = img.shape[0] * scaler, img.shape[1] * scaler
    result = cv2.resize(img, (w, h), interpolation=interpolation)
    return result

def upscale_large_img(img, scaler):
    upsampled_fraction_num = 0
    
    def upscale_img(_img):
        nonlocal upsampled_fraction_num
        h, w, c = _img.shape
        if h * w < PIXELS_LIMIT:
            upsampled_fraction_num += 1
            return sr.upsample(_img)
        
        if w < h:
            upper = np.zeros((scaler*h, scaler*w, c))
            below = np.zeros((scaler*h, scaler*w, c))

            upper[0:scaler*(h//2 + OVERLAP_HALF_LENGTH),:] = upscale_img(_img[0:(h//2 + OVERLAP_HALF_LENGTH),:])
            below[scaler*(h//2 - OVERLAP_HALF_LENGTH):scaler*h,:] = upscale_img(_img[(h//2 - OVERLAP_HALF_LENGTH):h,:])

            # h//2 - OVERLAP_HALF_LENGTH ~ h//2 + OVERLAP_HALF_LENGTH
            alpha = np.ones((scaler*h, scaler*w)) * np.arange(scaler * h).reshape(-1, 1)
            start = scaler * (h//2 - OVERLAP_HALF_LENGTH)
            end = scaler * (h//2 + OVERLAP_HALF_LENGTH)
            alpha = np.clip((alpha - start) / (end - start), 0, 1)

            alpha_3ch = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

            upper_f = upper.astype(np.float32)
            below_f = below.astype(np.float32)

            blended = upper_f * (1-alpha_3ch) + below_f * alpha_3ch
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            return blended
        else:
            left = np.zeros((scaler*h, scaler*w, c))
            right = np.zeros((scaler*h, scaler*w, c))
            
            left[:,0:scaler*(w//2 + OVERLAP_HALF_LENGTH)] = upscale_img(_img[:,0:(w//2 + OVERLAP_HALF_LENGTH)])
            right[:,scaler*(w//2 - OVERLAP_HALF_LENGTH):scaler*w] = upscale_img(_img[:,(w//2 - OVERLAP_HALF_LENGTH):w])

            # w//2 - OVERLAP_HALF_LENGTH ~ w//2 + OVERLAP_HALF_LENGTH
            alpha = np.ones((scaler*h, scaler*w)) * np.arange(scaler * w).reshape(1, -1)
            start = scaler * (w//2 - OVERLAP_HALF_LENGTH)
            end = scaler * (w//2 + OVERLAP_HALF_LENGTH)
            alpha = np.clip((alpha - start) / (end - start), 0, 1)
            
            alpha_3ch = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
            
            left_f = left.astype(np.float32)
            right_f = right.astype(np.float32)

            blended = left_f * (1-alpha_3ch) + right_f * alpha_3ch
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            return blended
        
    if scaler not in [2, 3, 4]:
        print(f"Invalid scaler value: {scaler}. Must be 2, 3 or 4.")
        return None
    if not cv2.cuda.getCudaEnabledDeviceCount():
        print("No CUDA-enabled GPU found.")
        return None

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    try:
        sr.readModel(f'models/EDSR_x{scaler}.pb')
    except Exception as e:
        print(f"Error reading model: {e}")
        return None

    # gpu acceleration
    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    sr.setModel('edsr', scaler)

    t1 = time.time()
    result = upscale_img(img)
    t2 = time.time()

    if upsampled_fraction_num > 1:
        print(f'upscaled after being divided into {upsampled_fraction_num} fragments.')
    else:
        print(f'upscaled without fraction')
    print(f'{t2-t1} sec taken')

    del sr
    gc.collect()
    return result

def _blend_images_with_contour_distance(A, B, contour, blend=BLEND_SINUSOIDAL):
    """
    거리 기반 알파 블렌딩으로 이미지 합성
    Args:
        A, B (np.ndarray): (h, w, 3) 원본 이미지
        contour (np.ndarray): 다각형 외곽선 (Nx1x2)
        blend (int): 블렌딩 가중치 방식
    Returns:
        np.ndarray: 합성 이미지
    """
    h, w = A.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    dist_transform = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=3)
    max_distance = np.max(dist_transform) * _DIST_CRIT_COEF
    if max_distance == 0:
        alpha = np.ones_like(dist_transform)
    else:
        v = 1 - dist_transform / max_distance
        if blend == BLEND_LINEAR:
            alpha = inter.np_linear(v, 0, 1)
        elif blend == BLEND_HERMIT_3:
            alpha = inter.np_hermit_3(v, 0, 1)
        elif blend == BLEND_HERMIT_5:
            alpha = inter.np_hermit_5(v, 0, 1)
        elif blend == BLEND_SINUSOIDAL:
            alpha = inter.np_sinusoidal(v, 0, 1)
        elif blend == BLEND_STEP:
            alpha = inter.np_unit_step(v, 0, 1)
        else:
            print("No such blend option")
            return None
    alpha[mask == 0] = 1
    alpha_3ch = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    A_f = A.astype(np.float32)
    B_f = B.astype(np.float32)
    blended = A_f * alpha_3ch + B_f * (1 - alpha_3ch)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended

def restore_img_mult_tgs(input_path, mrs3_mode, output_path=""):
    """
    압축 해제된 이미지/마스크/메타데이터 폴더에서 복원 이미지를 생성합니다.

    Args:
        input_path (str): 압축 해제 폴더 경로
        mrs3_mode (int): 업스케일 모드 (cv2.INTER_CUBIC 등, EDSR은 -1)
        output_path (str): 복원 이미지 저장 폴더 (미지정시 현재 폴더)
    Returns:
        None (결과 이미지는 파일로 저장)
    """
    if not os.path.exists(f'{input_path}/{downscaled_filename}.png'):
        print(f"Error loading image: {input_path}/{downscaled_filename}.png")
        return
    if not os.path.exists(f'{input_path}/{config_filename}.ini'):
        print(f'Error loading config: {input_path}/{config_filename}.ini')
        return
    if not os.path.exists(f'{input_path}/{roi_filename}0.png'):
        print(f'Error loading image: {input_path}/{roi_filename}0.png')
        return

    config = configparser.ConfigParser()
    config.read(f'{input_path}/{config_filename}.ini')
    target_num = int(config['DEFAULT']['NUMBER_OF_TARGETS'])
    scaler = int(config['DEFAULT']['SCALER'])

    if mrs3_mode == -1:
        upscaled = _upscale_by_edsr(f'{input_path}/{downscaled_filename}.png', scaler=scaler)
    else:
        upscaled = _upscale_by_resize(f'{input_path}/{downscaled_filename}.png', scaler=scaler, interpolation=mrs3_mode)
    restored = upscaled.copy()

    for i in range(target_num):
        y_from, y_to, x_from, x_to = int(config[f'{i}']['Y_FROM']), int(config[f'{i}']['Y_TO']), int(config[f'{i}']['X_FROM']), int(config[f'{i}']['X_TO'])
        roi = cv2.imread(f'{input_path}/{roi_filename}{i}.png')
        roi_mask = cv2.imread(f'{input_path}/{roi_binary_filename}{i}.png')

        bool_roi_mask_3ch = roi_mask > 0
        bool_roi_mask_1ch = np.all(roi_mask != [0, 0, 0], axis=2)
        bin_roi_mask = bool_roi_mask_1ch.astype(np.uint8) * 255

        combined_roi = np.where(bool_roi_mask_3ch, roi, upscaled[y_from:y_to, x_from:x_to])
        restored[y_from:y_to, x_from:x_to] = combined_roi

        contours, _ = cv2.findContours(bin_roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        restored[y_from:y_to, x_from:x_to] = _blend_images_with_contour_distance(
            upscaled[y_from:y_to, x_from:x_to],
            restored[y_from:y_to, x_from:x_to],
            contours[0],
            blend=BLEND_SINUSOIDAL
        )

    # 결과 저장
    if output_path == "":
        cv2.imwrite(f'{restored_filename}.png', restored)
        print(f"복원 이미지 저장 완료: {restored_filename}.png")
    else:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        out_file = os.path.join(output_path, f'{restored_filename}.png')
        cv2.imwrite(out_file, restored)
        print(f"복원 이미지 저장 완료: {out_file}")
