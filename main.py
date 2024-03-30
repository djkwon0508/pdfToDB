import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
from glob import glob
from pdf2image import convert_from_path
from tqdm import tqdm
import shutil


# 이미지에서 도형을 추출하여 이미지로 저장하는 함수
def extract_shapes_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 2:
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
                cv2.imwrite(f"{image_path}_shape{shape_count}.jpg", image)
                shape_count += 1
            elif len(approx) == 4:
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
                cv2.imwrite(f"{image_path}_shape{shape_count}.jpg", image)
                shape_count += 1
            elif len(approx) > 6:
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
                cv2.imwrite(f"{image_path}_shape{shape_count}.jpg", image)
                shape_count += 1


# 이미지에서 텍스트를 추출하는 함수
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    with open(f"{image_path}.txt", "w") as text_file:
        text_file.write(text)


# PDF를 이미지로 변환하고 각 이미지에서 도형을 추출하여 이미지로 저장하는 함수
def pdf_to_images_with_shapes(pdf_path, output_path, dpi=200):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        # 이전에 저장된 폴더 삭제
        shutil.rmtree(output_path)
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

            extract_shapes_from_image(image_path)
            extract_text_from_image(image_path)


if __name__ == "__main__":
    # PDF 파일을 이미지로 변환하고 도형을 추출하여 이미지로 저장합니다.
    pdf_path = "input.pdf"
    output_path = "output"
    pdf_to_images_with_shapes(pdf_path, output_path)
