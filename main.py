import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image
import os
from glob import glob
from pdf2image import convert_from_path
from tqdm import tqdm
import shutil


# 이미지에서 텍스트를 추출하는 함수
def extract_text_from_image(input_path, output_path):
    img1 = np.array(Image.open(input_path))
    text = pytesseract.image_to_string(img1, lang='kor+en+equ')
    # print(text)
    save_text_to_file(text, output_path)


# 이미지에서 텍스트의 정보를 추출하는 함수
def find_text_from_image(path):
    img = cv2.imread(path)
    text_data = pytesseract.image_to_data(img, lang='kor+equ', output_type=Output.DICTg)
    for iter in range(0, len(text_data["text"])):
        # Text Location
        x = text_data["left"][iter]
        y = text_data["top"][iter]
        # Text Weight
        w = text_data["width"][iter]
        # Text Height
        h = text_data["height"][iter]
        # Text Content
        txt = text_data["text"][iter]
        # Text Confidence Value
        conf = int(text_data["conf"][iter])

        if conf > 70:
            txt = "".join([c if ord(c) < 128 else "" for c in txt]).strip()
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(img, txt, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
            cv2.imshow('custom window name', img)
            cv2.waitKey(0)


def save_text_to_file(text, output_path):
    with open(output_path, 'w') as file:
        file.write(text)


# PDF를 이미지로 변환하고 각 이미지에서 도형을 추출하여 이미지로 저장하는 함수
def pdf_to_images_with_shapes(pdf_path, output_path, dpi=200):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        # 이전에 저장된 폴더 삭제
        shutil.rmtree(output_path)
        os.makedirs(output_path)

    for pdf_file in glob(pdf_path):
        # Set PDF Name
        pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]
        # Set PDF Dir
        pdf_output_path = os.path.join(output_path, pdf_name)
        if not os.path.exists(pdf_output_path):
            os.makedirs(pdf_output_path)

        images = convert_from_path(pdf_file, dpi=dpi)
        for page, image in tqdm(enumerate(images), desc=f"Converting {pdf_name}"):
            # Set Page Dir
            page_output_path = os.path.join(pdf_output_path, f"page_{page+1}")
            if not os.path.exists(page_output_path):
                os.makedirs(page_output_path)

            image_path = os.path.join(page_output_path, f"page_{page+1}.png")
            text_path = os.path.join(page_output_path, f"page_{page+1}.txt")
            image.save(image_path, "PNG")
            extract_text_from_image(image_path, text_path)
            #find_text_from_image(image_path)


def main():
    pdf_path = "*.pdf"
    output_path = "output"
    pdf_to_images_with_shapes(pdf_path, output_path, dpi=400)


if __name__ == "__main__":
    main()
