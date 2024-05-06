import cv2
from rembg import remove 
import numpy as np
from PIL import Image 

class BackgroundChanger:
    def __init__(self, input_image_path, new_background_path, output_image_path):
        self.input_image_path = input_image_path
        self.new_background_path = new_background_path
        self.output_image_path = output_image_path

    def remove_background(self):
        # 배경 제거할 이미지 불러오기
        input_image = Image.open(self.input_image_path)

        # 배경 제거하기
        removed_bg_image = remove(input_image)

        return removed_bg_image

    def add_background(self, removed_bg_image):
        # 새로운 배경 이미지 불러오기
        new_background = Image.open(self.new_background_path)

        # 배경 제거된 이미지와 새로운 배경 이미지 크기 맞추기
        new_background = new_background.resize(removed_bg_image.size)

        # 배경 제거된 이미지와 새로운 배경 이미지 합치기
        new_background.paste(removed_bg_image, (0,0), removed_bg_image)

        return new_background

    def save_output_image(self, output_image):
        # 합쳐진 이미지 저장
        output_image.save(self.output_image_path)

    def run(self):
        removed_bg_image = self.remove_background()
        output_image = self.add_background(removed_bg_image)
        self.save_output_image(output_image)


# 사용 예시
# background_changer = BackgroundChanger("coco_hollywood.jpg", "sky.jpg", "final_image.png")
# background_changer.run()
