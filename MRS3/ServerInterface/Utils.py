# 각종 유틸, 보조함수 등

"""
1. 웹 연결 후 압축 추가로 적용
2. 새로운 파일 확장자
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import os
import time
import struct
# import 

def compress_imgpresso(img_path, output_path):
    """
    img_path: absolute path of img
    output_path: absolute path of output folder
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless=new") # 헤드리스 모드

    prefs = {
        "download.default_directory": output_path,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }

    chrome_options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(options=chrome_options)
    wait = WebDriverWait(driver, 30)  # 대기 시간 30초

    try:
        # 1. 홈페이지 접속
        driver.get("https://imgpresso.co.kr")
        
        # 2. 이미지 업로드
        upload_input = wait.until(EC.presence_of_element_located(
            (By.XPATH, "//input[@type='file']")))
        upload_input.send_keys(img_path)
        
        # 3. 옵션 선택(압축비율)
        processing_complete = wait.until(EC.visibility_of_element_located(
            (By.XPATH, "//div[contains(@id, 'conv-area-upload')]")))

        option_xpath = "//div[@id='b-ip-profile-2']"
        option_element = wait.until(EC.element_to_be_clickable(
            (By.XPATH, option_xpath)))
        option_element.click()
        
        # 4. 다운로드 버튼 클릭
        confirm_btn = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//div[@id='b-ip-upload-ok']")))        
        confirm_btn.click()
        
        # 5. 처리 완료 대기
        processing_complete = wait.until(EC.visibility_of_element_located(
            (By.XPATH, "//div[contains(@id, 'conv-area-complete')]")))
        
        # 6. 다운로드 버튼 클릭
        download_btn = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//div[contains(@id, 'b-ip-download')]")))
        download_btn.click()
        
        # 7. 다운로드 완료 확인
        # TODO: 아래 코드에선 이미 다운로드된 png 파일에 의해 정상적으로 작동 안 됨 -> 새로운 png 만 인식하도록 수정할 필요 있음.
        start_time = time.time()
        while not any(fname.endswith('.png') for fname in os.listdir(output_path)):
            if time.time() - start_time > 60:
                raise TimeoutError("다운로드 시간 초과")
            time.sleep(1)
        
    finally:
        time.sleep(1) # 보조장치 -> 10초는 너무 길고, 줄일 필요 있음
        driver.quit()



    return

# 같은 폴더면 안 되는 현상 발생 -> 이미지 참조 중에 삭제 같은 이유라기엔 이미 업로드 끝내고 삭제하는거라 아닌거 같고
def compress_imgpresso_replace(img_path, output_path):
    """
    output_path 에 이미 같은 이름(basename)의 파일이 있으면 기존 파일 지우고 용량이 감소한 파일이 대체하도록 설정
    img_path: relative path of img
    output_path: relative path of output folder
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless=new") # 헤드리스 모드

    prefs = {
        "download.default_directory": os.path.join(os.getcwd(), output_path),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }

    chrome_options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(options=chrome_options)
    wait = WebDriverWait(driver, 10)  # 대기 시간 설정

    try:
        # 1. 홈페이지 접속
        driver.get("https://imgpresso.co.kr")
        
        # 2. 이미지 업로드
        upload_input = wait.until(EC.presence_of_element_located(
            (By.XPATH, "//input[@type='file']")))
        upload_input.send_keys(os.path.join(os.getcwd(), img_path))
        
        # 3. 옵션 선택(압축비율)
        processing_complete = wait.until(EC.visibility_of_element_located(
            (By.XPATH, "//div[contains(@id, 'conv-area-upload')]")))

        option_xpath = "//div[@id='b-ip-profile-2']"
        option_element = wait.until(EC.element_to_be_clickable(
            (By.XPATH, option_xpath)))
        option_element.click()
        
        # 4. 다운로드 버튼 클릭
        confirm_btn = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//div[@id='b-ip-upload-ok']")))        
        confirm_btn.click()
        
        # 5. 처리 완료 대기
        processing_complete = wait.until(EC.visibility_of_element_located(
            (By.XPATH, "//div[contains(@id, 'conv-area-complete')]")))
        
        # 5.5. 같은 이름의 파일이 있으면 삭제
        basename = os.path.basename(img_path)
        file_path = os.path.join(output_path, basename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

        # 6. 다운로드 버튼 클릭
        download_btn = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//div[contains(@id, 'b-ip-download')]")))
        download_btn.click()
        
        # 7. 다운로드 완료 확인
        # TODO: 아래 코드에선 이미 다운로드된 png 파일에 의해 정상적으로 작동 안 됨 -> 새로운 png 만 인식하도록 수정할 필요 있음.
        start_time = time.time()
        while not any(fname.endswith('.png') for fname in os.listdir(output_path)):
            if time.time() - start_time > 60:
                raise TimeoutError("다운로드 시간 초과")
            time.sleep(1)
        
    finally:
        time.sleep(1) # 보조장치 -> 10초는 너무 길고, 줄일 필요 있음. 1초도 좀 긴가
        driver.quit()



    return


def pack_files(output_file: str, input_files: list):
    """
    여러 파일을 하나의 사용자 정의 패키지(.pkg)로 묶습니다.
    
    :param output_file: 출력 파일 경로 (예: 'output.pkg')
    :param input_files: 패키징할 파일 목록 (예: ['image.png', 'config.ini'])
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
            
            # 파일명 길이 및 데이터 기록
            encoded_name = file_name.encode('utf-8')
            f_out.write(struct.pack('I', len(encoded_name)))  # 파일명 길이 (4바이트)
            f_out.write(encoded_name)                         # 파일명 데이터
            
            # 파일 내용 기록
            f_out.write(struct.pack('I', len(file_data)))     # 파일 크기 (4바이트)
            f_out.write(file_data)                            # 파일 내용

def unpack_files(input_file: str, output_dir: str):
    """
    패키지 파일(.pkg)에서 원본 파일을 추출합니다.
    
    :param input_file: 입력 파일 경로 (예: 'output.pkg')
    :param output_dir: 출력 디렉토리 경로 (예: 'unpacked')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_file, 'rb') as f_in:
        # 파일 개수 읽기
        num_files = struct.unpack('I', f_in.read(4))[0]
        
        for _ in range(num_files):
            # 파일명 추출
            name_len = struct.unpack('I', f_in.read(4))[0]
            file_name = f_in.read(name_len).decode('utf-8')
            
            # 파일 내용 추출
            data_len = struct.unpack('I', f_in.read(4))[0]
            file_data = f_in.read(data_len)
            
            # 파일 저장
            output_path = os.path.join(output_dir, file_name)
            with open(output_path, 'wb') as f_out:
                f_out.write(file_data)



