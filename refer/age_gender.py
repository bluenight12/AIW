import openvino as ov
import cv2
import numpy as np
import ipywidgets as widgets

# from typing import Tuple
# import os
# import sys


# FaceDetection 클래스는 얼굴 감지를 담당합니다. 이 클래스는 OpenVINO 모델을 사용하여 이미지에서 얼굴을 감지하고, 감지된 얼굴을 잘라내는 기능을 제공합니다.
class FaceDetection:
    def __init__(self, model_xml, model_bin, device="CPU"):
        # 모델 파일의 경로와 사용할 디바이스를 입력으로 받아 초기화합니다.
        self.model_xml = model_xml
        self.model_bin = model_bin

        self.core = ov.Core()
        self.device = widgets.Dropdown(
            options=self.core.available_devices + ["AUTO"],
            value='AUTO',
            description='Device:',
            disabled=False,
        )
        self.model = self.core.read_model(model=self.model_xml)         # 모델을 로드합니다.
        self.compiled_model = self.core.compile_model(      
            model=self.model, device_name=self.device.value)            # 모델을 실행할 디바이스에 맞게 최적화합니다.
        self.input_keys = self.compiled_model.input(0)                  # 모델의 입력 키를 가져옵니다.
        self.output_keys = self.compiled_model.output(0)                # 각각 나이와 성별의 출력 키를 가져옵니다.
        self.height, self.width = list(self.input_keys.shape)[2:]       # 모델의 입력 이미지 크기를 설정 합니다.

    def preprocess_image(self, image):
        # 이미지를 모델 입력에 맞게 전처리합니다.
        resized_image = cv2.resize(image, (self.width, self.height))
        input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
        return input_image

    def detect_faces(self, processed_image):
        # 전처리된 이미지에서 얼굴을 감지합니다.
        results = self.compiled_model([processed_image])[self.output_keys]
        results = np.squeeze(results, (0, 1))
        results = results[~np.all(results == 0, axis=1)]
        return results

    def crop_faces(self, bgr_image, results, threshold=0.6):
        # 감지된 얼굴을 이미지에서 잘라냅니다.
        (real_y, real_x) = bgr_image.shape[:2]
        results = results[:, :]
        cropped_faces = []
        for box in results:
            conf = box[2]
            if conf > threshold:
                (x_min, y_min, x_max, y_max) = [
                    int(corner_position * real_y) if idx % 2
                    else int(corner_position * real_x)
                    for idx, corner_position in enumerate(box[3:])
                ]
                cropped_face = bgr_image[y_min:y_max, x_min:x_max]
                if cropped_face.shape[0] != 0 and cropped_face.shape[1] != 0:
                    cropped_faces.append(cropped_face)

        return cropped_faces

    def run(self, image):
        # 이미지를 입력받아 감지된 이미지들을 리턴합니다.
        processed_image = self.preprocess_image(image)
        faces = self.detect_faces(processed_image)
        face_images = self.crop_faces(image, faces)
        return face_images


class AgeGenderPrediction:          # AgeGenderPrediction 클래스는 성별 및 연령 예측을 담당합니다. 이 클래스는 OpenVINO 모델을 사용하여 감지된 얼굴의 성별과 연령을 예측합니다.
    def __init__(self, model_xml, model_bin, device="CPU"):
        # 모델 파일의 경로와 사용할 디바이스를 입력으로 받아 초기화합니다.
        self.model_xml = model_xml
        self.model_bin = model_bin
        self.core = ov.Core()
        self.device = widgets.Dropdown(
            options=self.core.available_devices + ["AUTO"],
            value='AUTO',
            description='Device:',
            disabled=False,
        )
        self.model = self.core.read_model(model=self.model_xml)     # 모델을 로드합니다.
        self.compiled_model = self.core.compile_model(          
            model=self.model, device_name=self.device.value)        # 모델을 실행할 디바이스에 맞게 최적화합니다.
        self.input_keys = self.compiled_model.input(0)              # 모델의 입력 키를 가져옵니다.
        self.age_output_key = self.compiled_model.output(1)         # 각각 나이와 성별의 출력 키를 가져옵니다.
        self.gender_output_key = self.compiled_model.output(0)
        self.height, self.width = list(self.input_keys.shape)[2:]   # 모델의 입력 이미지 크기를 설정 합니다.

    def preprocess_image(self, image):
        # 이미지를 모델 입력에 맞게 전처리합니다.
        resized_image = cv2.resize(image, (self.width, self.height))
        input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
        return input_image

    def predict_age_gender(self, processed_image):
        # 전처리된 이미지에서 성별과 연령을 예측합니다.
        results = self.compiled_model([processed_image])
        age = results[self.age_output_key][0][0][0][0] * 100
        gender = results[self.gender_output_key]
        return age, 'Male' if gender[0, 0, 0, 0] < gender[0, 1, 0, 0] else 'Female'

    def run(self, face_images):
        # 감지된 이미지들을 받아 각각의 연령과 성별을 리턴합니다.
        # ages = []
        # genders = []
        age = None
        gender = None
        i = 0
        for face in face_images:
            i += 1
            processed_face = self.preprocess_image(face)
            age, gender = self.predict_age_gender(processed_face)
            if i == 1:
                break
            # ages.append(age)
            # genders.append(gender)
            # print(f"Detected {gender} face, age {age}")
        return age, gender