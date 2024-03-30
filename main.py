import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
from glob import glob
from pdf2image import convert_from_path
from tqdm import tqdm


def extract_text_from_image(image_path):
    # 이미지 불러오기
    img = Image.open(image_path)

    # 이미지에서 텍스트 추출
    text = pytesseract.image_to_string(img, lang='kor+eng')  # 한국어와 영어 언어 설정

    return text


def extract_shapes_and_functions(image_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 엣지 검출
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 컨투어(윤곽선) 검출
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 도형 및 함수 추출
    shapes = []
    functions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # 일정 크기 이상의 컨투어만 고려
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # 직선인지 확인
            if len(approx) == 2:
                shapes.append("Line")
            # 사각형인지 확인
            elif len(approx) == 4:
                shapes.append("Rectangle")
            # 원 또는 반원인지 확인
            elif len(approx) > 6:
                # 원인지 확인
                circle = cv2.minEnclosingCircle(contour)
                if circle[1] > 0:
                    center, radius = circle
                    area_ratio = area / (np.pi * radius ** 2)
                    if 0.5 < area_ratio < 1.5:  # 반원의 비율은 약 1에 가깝다
                        shapes.append("Semicircle")
                    else:
                        shapes.append("Circle")
                # n각형 또는 n면체인지 확인
                else:
                    # n각형 및 n면체 검출 코드 추가
                    if len(approx) > 4:
                        shapes.append("Polygon")
                    else:
                        shapes.append("Unknown")

            # 함수 추출 및 저장
            text = extract_text_from_image(image_path)
            # 함수를 추출하는 코드를 추가해주세요.
            functions.append(text)

    return shapes, functions


def pdf_to_image(pdf_path, output_path, dpi=200):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pdf_files = glob(pdf_path)
    for pdf_file in pdf_files:
        pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]
        pdf_output_path = os.path.join(output_path, pdf_name)
        if not os.path.exists(pdf_output_path):
            os.makedirs(pdf_output_path)

        images = convert_from_path(pdf_file, dpi=dpi)
        for i, image in tqdm(enumerate(images), desc=f"Converting {pdf_name}"):
            page_output_path = os.path.join(pdf_output_path, f"page_{i + 1}")
            if not os.path.exists(page_output_path):
                os.makedirs(page_output_path)

            image_path = os.path.join(page_output_path, f"page_{i + 1}.jpg")
            image.save(image_path, "JPEG")

            # 이미지에서 텍스트 및 함수 추출
            text = extract_text_from_image(image_path)
            with open(os.path.join(page_output_path, f"page_{i + 1}_text.txt"), "w", encoding="utf-8") as text_file:
                text_file.write(text)

            # 이미지에서 도형 및 함수 추출
            shapes, functions = extract_shapes_and_functions(image_path)
            with open(os.path.join(page_output_path, f"page_{i + 1}_shapes.txt"), "w") as shapes_file:
                shapes_file.write("\n".join(shapes))
            with open(os.path.join(page_output_path, f"page_{i + 1}_functions.txt"), "w") as functions_file:
                functions_file.write("\n".join(func for func in functions))


if __name__ == "__main__":
    # PDF 파일 경로
    pdf_path = "*.pdf"

    # 이미지로 변환한 결과를 저장할 폴더 경로
    output_path = "output_images"

    # PDF를 이미지로 변환하고 도형, 함수 및 함수 그래프 추출하여 저장
    pdf_to_image(pdf_path, output_path)
