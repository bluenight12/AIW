import os
import cv2
import time
import gdown
import torch
import requests
import numpy as np
import openvino as ov
import ipywidgets as widgets
import matplotlib.pyplot as plt
from pathlib import Path
from collections import namedtuple
from IPython.display import HTML, FileLink, display
from notebook_utils import load_image, download_file


def Mix_img():
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 상위 디렉토리 추가

    from models.u2net import U2NET, U2NETP

    # Import local modules
    MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')  # 모델 디렉토리 상대 경로로 수정
    
    if not Path("./notebook_utils.py").exists():
        # Fetch `notebook_utils` module

        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py", directory=MODEL_DIR  # 모델 디렉토리로 수정
        )

        open("notebook_utils.py", "w").write(r.text)

    if not Path("./model/u2net.py").exists():
        download_file(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/vision-background-removal/model/u2net.py", directory=MODEL_DIR  # 모델 디렉토리로 수정
    )


    model_config = namedtuple("ModelConfig", ["name", "url", "model", "model_args"])

    u2net_lite = model_config(
        name="u2net_lite",
        url="https://drive.google.com/uc?id=1W8E4FHIlTVstfRkYmNOjbr0VDXTZm0jD",
        model=U2NETP,
        model_args=(),
    )
    u2net = model_config(
        name="u2net",
        url="https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
        model=U2NET,
        model_args=(3, 1),
    )
    u2net_human_seg = model_config(
        name="u2net_human_seg",
        url="https://drive.google.com/uc?id=1m_Kgs91b21gayc2XLW0ou8yugAIadWVP",
        model=U2NET,
        model_args=(3, 1),
    )

    # Set u2net_model to one of the three configurations listed above.
    u2net_model = u2net_lite

    # 모델 파일의 경로 지정
    model_path = Path(os.path.join(MODEL_DIR, 'u2net_lite', 'u2net_lite.pth'))  # 모델 디렉토리 상대 경로로 수정

    if not model_path.exists():

        os.makedirs(name=model_path.parent, exist_ok=True)
        print("Start downloading model weights file... ")
        with open(model_path, "wb") as model_file:
            gdown.download(url=u2net_model.url, output=model_file)
            print(f"Model weights have been downloaded to {model_path}")

    # Load the model.
    net = u2net_model.model(*u2net_model.model_args)
    net.eval()

    # Load the weights.
    print(f"Loading model weights from: '{model_path}'")
    net.load_state_dict(state_dict=torch.load(model_path, map_location="cpu"))

    model_ir = ov.convert_model(net, example_input=torch.zeros((1, 3, 512, 512)), input=([1, 3, 512, 512]))

    # Load image path
    IMAGE_URI = os.path.join(os.path.dirname(__file__), '..', 'img', 'removed_background.jpg')  # 이미지 경로 상대 경로로 수정

    input_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
    input_scale = np.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)

    image = cv2.cvtColor(
        src=load_image(IMAGE_URI),
        code=cv2.COLOR_BGR2RGB,
    )

    resized_image = cv2.resize(src=image, dsize=(512, 512))
    # Convert the image shape to a shape and a data type expected by the network
    # for OpenVINO IR model: (1, 3, 512, 512).
    input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

    input_image = (input_image - input_mean) / input_scale

    core = ov.Core()
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="AUTO",
        description="Device:",
        disabled=False,
    )

    device

    core = ov.Core()
    # Load the network to OpenVINO Runtime.
    compiled_model_ir = core.compile_model(model=model_ir, device_name=device.value)
    # Get the names of input and output layers.
    input_layer_ir = compiled_model_ir.input(0)
    output_layer_ir = compiled_model_ir.output(0)

    # Do inference on the input image.
    start_time = time.perf_counter()
    result = compiled_model_ir([input_image])[output_layer_ir]
    end_time = time.perf_counter()
    print(f"Inference finished. Inference time: {end_time-start_time:.3f} seconds, " f"FPS: {1/(end_time-start_time):.2f}.")

    # Resize the network result to the image shape and round the values
    # to 0 (background) and 1 (foreground).
    # The network result has (1,1,512,512) shape. The `np.squeeze` function converts this to (512, 512).
    resized_result = np.rint(cv2.resize(src=np.squeeze(result), dsize=(image.shape[1], image.shape[0]))).astype(np.uint8)

    # Create a copy of the image and set all background values to 255 (white).
    bg_removed_result = image.copy()
    bg_removed_result[resized_result == 0] = 255

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))
    ax[0].imshow(image)
    ax[1].imshow(resized_result, cmap="gray")
    ax[2].imshow(bg_removed_result)
    for a in ax:
        a.axis("off")

    # Load BG_image path
    BACKGROUND_FILE = os.path.join(os.path.dirname(__file__), '..', 'img', 'bg_img.jpg')  # 배경 이미지 경로 상대 경로로 수정
    
    background_image = cv2.cvtColor(src=load_image(BACKGROUND_FILE), code=cv2.COLOR_BGR2RGB)
    background_image = cv2.resize(src=background_image, dsize=(image.shape[1], image.shape[0]))

    # Set all the foreground pixels from the result to 0
    # in the background image and add the image with the background removed.
    background_image[resized_result == 1] = 0
    new_image = background_image + bg_removed_result

    # Save the generated image.
    new_image_path = os.path.join(os.path.dirname(__file__), '..', 'img', 'new_img.png')  # 새로운 이미지 경로 상대 경로로 수정
    cv2.imwrite(filename=str(new_image_path), img=cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))

   
Mix_img()
