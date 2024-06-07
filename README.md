# AIW
의류 매장에 등록되어 있는 옷들을 바탕으로 옷을 데이터베이스에 넣고, LLM을 통해 추천을 받아 그와 비슷한 옷을 입혀주고 마지막으로 배경을 바꾸고 싶다면 가상 의류 시작 사진에 배경을 바꿔 보여주는 의류 매장 AI 키오스크

## 프로젝트 소개
![그림18](https://github.com/bluenight12/AIW/assets/154478957/744b96c2-16f9-440e-842e-3213e350e0c1)

## 프로젝트 프로세스
![그림19](https://github.com/bluenight12/AIW/assets/154478957/72b55fba-a2c5-400c-9f79-0db6551a51a7)
![그림20](https://github.com/bluenight12/AIW/assets/154478957/375262eb-35f3-44a9-a735-8e8993bc3437)


## 기능 

### Computer Vision
* [face-detection](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/face-detection-retail-0005) : 사용자의 연령 및 성별 인식을 정확하게 할 수 있도록 얼굴을 크롭하는 기능
* [age-gender]( https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/age-gender-recognition-retail-0013) : 사용자의 연령 및 성별을 인식하는 기능
* [tflite-selfie-segmentation](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/tflite-selfie-segmentation/tflite-selfie-segmentation.ipynb): 배경, 헤어, 얼굴, 의상, 신체, 기타 로 이미지의 마스크를 생성하는 기능
* [mediapie_pose_landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker?hl=ko): 사진 촬영시 자세 제안 기능 

### llm을 통한 의류 추천 
일반 의류 db에 다양한 정보를 openai를 통해 임베딩하여 chromadb에 저장하였습니다. 그리고 음성인식을 통해 텍스트를 입력받으면 그것이 쿼리가 되고 퀴리를 임베딩하여 서로의 유사도 검사를 통해 의상이 추천됩니다.
![그림08](https://github.com/bluenight12/AIW/assets/154478957/4454a462-c12c-4baa-8c7f-0a3b2166cae8)

 

### stable-diffusion-webui을 이용한 가상 의류 시착

촬영된 사진과 마스크된 이미지를 통해 stable-diffusion-webui-api를 통해 의류 시착 이미지를 생성합니다. 이때 LLM으로 추천된 의상의 prompt와 Fine-tuning된 LoRa 모델을 통해 의류를 고정합니다. 추가로 controlnet을 사용해 생성된 이미지의 자세를 고정시킵니다. 

### 배경생성 

생성된 시착이미지를 마스크된 이미지를 통해 배경을 생성합니다.

---
## Prerequite
### 가상환경 설정 및 requirements 설치
```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### stable-diffusion-webui 세팅

* [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 설치
* [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) 설치
* [controlnet-openpose](https://github.com/Mikubill/sd-webui-controlnet/wiki/Model-download) 모델 설치
* webui-user.bat(windows) , webui-user.sh(linux) set COMMANDLINE_ARGS 추가 ```--listen --share --api```
* webui 실행
```
Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://xxxxxxxxxxxxx.gradio.live
```
### refer/webui_api.py 링크 세팅
각자 상황에 맞는 webui url을 입력
```
self.webui_server_url = 'https://xxxxxxx.gradio.live'
```
### pages/voice.py openai-api 세팅

pages/voice.py의 api_key에 자신의 api코드를 입력
```
api_key = 'code'
```



---
## Steps to run

```shell
cd ~/xxxx
source .venv/bin/activate

cd /path/to/repo/xxx/
streamlit run main.py
```

