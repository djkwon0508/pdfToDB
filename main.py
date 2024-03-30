import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
from glob import glob
from pdf2image import convert_from_path


def extract_text_from_image(image_path):
    # 이미지 불러오기
    img = Image.open(image_path)

    # 이미지에서 텍스트 추출
    text = pytesseract.image_to_string(img, lang='kor+eng')

    return text


def extract_shapes_from_image(image_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 엣지 검출
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 직선 검출
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    # 직선들을 기반으로 사각형 추출
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image


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
        for i, image in enumerate(images):
            page_output_path = os.path.join(pdf_output_path, f"page_{i + 1}")
            if not os.path.exists(page_output_path):
                os.makedirs(page_output_path)

            image_path = os.path.join(page_output_path, f"page_{i + 1}.jpg")
            image.save(image_path, "JPEG")

            # 이미지에서 텍스트 추출
            extracted_text = extract_text_from_image(image_path)
            with open(os.path.join(page_output_path, f"page_{i + 1}_text.txt"), "w", encoding="utf-8") as text_file:
                text_file.write(extracted_text)

            # 이미지에서 도형 추출
            extracted_image = extract_shapes_from_image(image_path)
            cv2.imwrite(os.path.join(page_output_path, f"page_{i + 1}_shapes.jpg"), extracted_image)


if __name__ == "__main__":
    # PDF 파일 경로
    pdf_path = "*.pdf"

    # 이미지로 변환한 결과를 저장할 폴더 경로
    output_path = "output_images"

    # PDF를 이미지로 변환하고 도형 및 텍스트 추출하여 저장
    pdf_to_image(pdf_path, output_path)
