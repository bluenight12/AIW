#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이 파이썬 스크립트는 webui의 AIP를 활용하여 webui의 url만 있다면 외부에서 실행시킬수있는
파이썬 코드입니다.

아래의 깃허브의 내용을 참고하여 만들었습니다.
https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API

위 깃허브 내용에서는 urllib 라이브러를 사용하였지만 현재는 작동하지 않아
urllib3로 변경하여 해결하였습니다.

추가로 컨트롤넷을 활용할경우
https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/7784
아래의 깃허브의 내용을 보면 컨트롤넷도 활용가능할것으로 보이며 추후 추가 예정입니다.

"""
__author__ = "song-si-kyeong"
__date__ = "2024-05-04"

from datetime import datetime
import urllib3
import base64
import json
import time
import os
# import urllib.error
# import urllib.request


class Create_image :
    def __init__(self):
        self.webui_server_url = 'http://127.0.0.1:7860'
        self.http = urllib3.PoolManager()
        self.out_dir = 'api_out'
        self.out_dir_t2i = os.path.join(self.out_dir, 'txt2img')
        self.out_dir_i2i = os.path.join(self.out_dir, 'img2img')
        os.makedirs(self.out_dir_t2i, exist_ok=True)
        os.makedirs(self.out_dir_i2i, exist_ok=True)


    def timestamp(self):
        return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


    def encode_file_to_base64(self,path):
        with open(path, 'rb') as file:
            return base64.b64encode(file.read()).decode('utf-8')


    def decode_and_save_base64(self ,base64_str, save_path):
        with open(save_path, "wb") as file:
            file.write(base64.b64decode(base64_str))

    def call_api(self, api_endpoint, **payload):
        data = json.dumps(payload).encode('utf-8')

        response = self.http.request(
            method='POST',
            url=f'{self.webui_server_url}/{api_endpoint}',
            headers={'Content-Type': 'application/json'},
            body=data,
        )

        return json.loads(response.data.decode('utf-8'))



    def call_txt2img_api(self, **payload):
        response = self.call_api('sdapi/v1/txt2img', **payload)
        for index, image in enumerate(response.get('images',[])):
            save_path = os.path.join(self.out_dir_t2i, f'txt2img-{self.timestamp()}-{index}.png')
            self.decode_and_save_base64(image, save_path)


    def call_img2img_api(self, **payload):
        response = self.call_api('sdapi/v1/img2img', **payload)
        print("종료")
        for index, image in enumerate(response.get('images')):
            save_path = os.path.join(self.out_dir_i2i, f'img2img-{self.timestamp()}-{index}.png')
            self.decode_and_save_base64(image, save_path)


    def i2i(self, input_img_path, mask_img_path,prompt_txt ):

        input_img = input_img_path
        init_images = [
            self.encode_file_to_base64(input_img),
            # encode_file_to_base64(r"B:\path\to\img_2.png"),
            # "https://image.can/also/be/a/http/url.png",
        ]
        prom = prompt_txt
        batch_size = 1                      # 이미지 생성 갯수
        payload = {
            "prompt": prom,
            "negative_prompt": "(worst quality, greyscale), ac_neg2, zip2d_neg, ziprealism_neg, watermark, username, signature, text, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, bad feet, extra fingers, mutated hands, poorly drawn hands, bad proportions, extra limbs, disfigured, bad anatomy, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, fused fingers, too many fingers, long neck",
            "seed": -1,                         ## 시드 난수로
            "steps": 40,
            "width": 480,
            "height": 640,
            # "image_cfg_scale": 0.5,
            "denoising_strength": 0.7,          ## 이수치를 줄이면 기존이미지와 많이 멀어짐 .
            "n_iter": 1,
            "init_images": init_images,
            "batch_size": batch_size if len(init_images) == 1 else len(init_images),
            "cfg_scale": 7,
            "sampler_name": "DPM++ 2M",  ##  샘플링 방법을 결정하는 설정입니다.
            "scheduler": "Karras",  ## 학습 스케줄러를 설정하는 부분입니다.
            "mask": self.encode_file_to_base64(mask_img_path),
            "inpaint_full_res_padding": 50,  ## 인페인팅 패딩을 결정하는 설정입니다.

        }
        self.call_img2img_api(**payload)
        #     {
        #   "prompt": "",               ## 이미지 생성에 사용되는 초기 텍스트 프롬프트입니다.
        #   "negative_prompt": "",      ## 이미지 생성에 사용되는 부정적 프롬프트입니다.
        #   "styles": [                 ## 이미지 스타일을 결정하는데 사용되는 스타일 설정입니다.
        #     "string"
        #   ],
        #   "seed": -1,                 ## 난수 생성기의 시드값입니다. 이 값이 같으면 동일한 결과를 생성합니다.
        #   "subseed": -1,
        #   "subseed_strength": 0,
        #   "seed_resize_from_h": -1,
        #   "seed_resize_from_w": -1,


        #   "sampler_name": "string",   ##  샘플링 방법을 결정하는 설정입니다.
        #   "scheduler": "string",      ## 학습 스케줄러를 설정하는 부분입니다.
        #   "batch_size": 1,            ## 한 번에 처리하는 데이터의 양을 결정하는 배치 크기입니다.
        #   "n_iter": 1,                ## 반복 횟수를 결정하는 설정입니다.
        #   "steps": 50,                ## 각 반복에서 수행하는 스텝 수를 결정하는 설정입니다.
        #   "cfg_scale": 7,             ## 설정의 스케일을 결정하는 값입니다.
        #   "width": 512,
        #   "height": 512,

        #   "restore_faces": true,          ##얼굴 복원 기능을 활성화하는 설정입니다.
        #   "tiling": true,                 ## 타일링 기능을 활성화하는 설정입니다.
        #   "do_not_save_samples": false,   ## 샘플 및 그리드 저장을 비활성화하는 설정입니다.
        #   "do_not_save_grid": false,

        #   "eta": 0,                       ## 학습률을 결정하는 설정입니다.
        #   "denoising_strength": 0.75,     ## 노이즈 제거 강도를 결정하는 설정입니다.

        #   "s_min_uncond": 0,              ##샘플링과 관련된 추가 설정입니다.
        #   "s_churn": 0,
        #   "s_tmax": 0,
        #   "s_tmin": 0,
        #   "s_noise": 0,

        #   "override_settings": {},         ## 기본 설정을 임시로 덮어쓰는 설정입니다.
        #   "override_settings_restore_afterwards": true,

        #   "refiner_checkpoint": "string",         ## 리파이너 네트워크와 관련된 설정입니다.
        #   "refiner_switch_at": 0,
        #   "disable_extra_networks": false,
        #   "firstpass_image": "string",            ## 첫 번째 패스에서 사용되는 이미지를 지정하는 설정입니다.

        #   "comments": {},                         ## 사용자가 추가할 수 있는 주석입니다.
        #   "init_images": [
        #     "string"
        #   ],
        #   "resize_mode": 0,
        #   "image_cfg_scale": 0,
        #   "mask": "string",                   #  마스크를 적용할 영역을 결정하는 설정입니다. 마스크가 적용된 영역만 이미지 생성에 영향을 받습니다.
        #   "mask_blur_x": 4,                   #  마스크의 테두리를 흐리게 하는 설정입니다. 값이 클수록 더 흐릿해집니다.
        #   "mask_blur_y": 4,
        #   "mask_blur": 0,
        #   "mask_round": true,
        #   "inpainting_fill": 0,               ## 마스크 영역을 채우는 방법을 결정하는 설정입니다.
        #   "inpaint_full_res": true,           ## 전체 해상도에서 인페인팅을 수행할지 결정하는 설정입니다.
        #   "inpaint_full_res_padding": 0,      ## 인페인팅 패딩을 결정하는 설정입니다.
        #   "inpainting_mask_invert": 0,        ## 마스크 반전을 결정하는 설정입니다.
        #   "initial_noise_multiplier": 0,      ## 초기 노이즈의 배율을 결정하는 설정입니다.
        #   "latent_mask": "string",            ## 잠재 공간에서 마스크를 적용할 영역을 결정하는 설정입니다.
        #   "force_task_id": "string",          ## 작업 ID를 강제로 설정하는 설정입니다.
        #   "sampler_index": "Euler",           ## 샘플러의 인덱스를 결정하는 설정입니다.
        #   "include_init_images": false,       ## 초기 이미지를 결과에 포함시킬지 결정하는 설정입니다.

        #   "script_name": "string",            ## 스크립트 이름과 인수를 설정하는 부분입니다.
        #   "script_args": [],

        #   "send_images": true,
        #   "save_images": false,
        #   "alwayson_scripts": {},             ## 항상 실행되는 스크립트를 설정하는 부분입니다.
        #   "infotext": "string"                ## 추가 정보 텍스트를 설정하는 부분입니다.
        # }
        # if len(init_images) > 1 then batch_size should be == len(init_images)
        # else if len(init_images) == 1 then batch_size can be any value int >= 1


        # there exist a useful extension that allows converting of webui calls to api payload
        # particularly useful when you wish setup arguments of extensions and scripts
        # https://github.com/huchenlei/sd-webui-api-payload-display
    def t2i(self, prompt):
        payload = {
            "prompt": prompt,  # extra networks also in prompts
            "negative_prompt": "",
            "seed": -1,
            "steps": 20,
            "width": 480,
            "height": 640,
            "cfg_scale": 7,
            "sampler_name": "DPM++ 2M",  ##  샘플링 방법을 결정하는 설정입니다.
            "scheduler": "Karras",  ## 학습 스케줄러를 설정하는 부분입니다.
            "n_iter": 1,
            "batch_size": 1,

            # example args for x/y/z plot
            # "script_name": "x/y/z plot",
            # "script_args": [
            #     1,
            #     "10,20",
            #     [],
            #     0,
            #     "",
            #     [],
            #     0,
            #     "",
            #     [],
            #     True,
            #     True,
            #     False,
            #     False,
            #     0,
            #     False
            # ],

            # example args for Refiner and ControlNet
            # "alwayson_scripts": {
            #     "ControlNet": {
            #         "args": [
            #             {
            #                 "batch_images": "",
            #                 "control_mode": "Balanced",
            #                 "enabled": True,
            #                 "guidance_end": 1,
            #                 "guidance_start": 0,
            #                 "image": {
            #                     "image": encode_file_to_base64(r"B:\path\to\control\img.png"),
            #                     "mask": None  # base64, None when not need
            #                 },
            #                 "input_mode": "simple",
            #                 "is_ui": True,
            #                 "loopback": False,
            #                 "low_vram": False,
            #                 "model": "control_v11p_sd15_canny [d14c016b]",
            #                 "module": "canny",
            #                 "output_dir": "",
            #                 "pixel_perfect": False,
            #                 "processor_res": 512,
            #                 "resize_mode": "Crop and Resize",
            #                 "threshold_a": 100,
            #                 "threshold_b": 200,
            #                 "weight": 1
            #             }
            #         ]
            #     },
            #     "Refiner": {
            #         "args": [
            #             True,
            #             "sd_xl_refiner_1.0",
            #             0.5
            #         ]
            #     }
            # },
            # "enable_hr": True,
            # "hr_upscaler": "R-ESRGAN 4x+ Anime6B",
            # "hr_scale": 2,
            # "denoising_strength": 0.5,
            # "styles": ['style 1', 'style 2'],
            # "override_settings": {
            #     'sd_model_checkpoint': "sd_xl_base_1.0",  # this can use to switch sd model
            # },
        }
        self.call_txt2img_api(**payload)
####
# test

# aa = Create_image()
#
# aa.i2i(".\\1.jpg",".\\1.png","t-shirt")