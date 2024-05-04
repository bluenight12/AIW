import requests

r = requests.get(
    url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py", "w").write(r.text)
from pathlib import Path
from notebook_utils import download_file

tflite_model_path = Path("selfie_multiclass_256x256.tflite")
tflite_model_url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"

download_file(tflite_model_url, tflite_model_path)

import openvino as ov

core = ov.Core()

ir_model_path = tflite_model_path.with_suffix(".xml")

if not ir_model_path.exists():
    ov_model = ov.convert_model(tflite_model_path)
    ov.save_model(ov_model, ir_model_path)
else:
    ov_model = core.read_model(ir_model_path)

print(f"Model input info: {ov_model.inputs}")

print(f"Model output info: {ov_model.outputs}")

import ipywidgets as widgets

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value="AUTO",
    description="Device:",
    disabled=False,
)

device

compiled_model = core.compile_model(ov_model, device.value)

import cv2
import numpy as np
from notebook_utils import load_image

# Read input image and convert it to RGB
test_image_url = "C:\Users\admin\Documents\image.png"
img = load_image(test_image_url)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Preprocessing helper function
def resize_and_pad(image: np.ndarray, height: int = 256, width: int = 256):
    """
    Input preprocessing function, takes input image in np.ndarray format,
    resizes it to fit specified height and width with preserving aspect ratio
    and adds padding on bottom or right side to complete target height x width rectangle.

    Parameters:
      image (np.ndarray): input image in np.ndarray format
      height (int, *optional*, 256): target height
      width (int, *optional*, 256): target width
    Returns:
      padded_img (np.ndarray): processed image
      padding_info (Tuple[int, int]): information about padding size, required for postprocessing
    """
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (width, np.floor(h / (w / width)).astype(int)))
    else:
        img = cv2.resize(image, (np.floor(w / (h / height)).astype(int), height))

    r_h, r_w = img.shape[:2]
    right_padding = width - r_w
    bottom_padding = height - r_h
    padded_img = cv2.copyMakeBorder(img, 0, bottom_padding, 0, right_padding, cv2.BORDER_CONSTANT)
    return padded_img, (bottom_padding, right_padding)


# Apply preprocessig step - resize and pad input image
padded_img, pad_info = resize_and_pad(np.array(img))

# Convert input data from uint8 [0, 255] to float32 [0, 1] range and add batch dimension
normalized_img = np.expand_dims(padded_img.astype(np.float32) / 255, 0)

out = compiled_model(normalized_img)[0]

from typing import Tuple
from notebook_utils import segmentation_map_to_image, SegmentationMap, Label
import matplotlib.pyplot as plt


# helper for visualization segmentation labels
labels = [
    Label(index=0, color=(192, 192, 192), name="background"),
    #Label(index=1, color=(128, 0, 0), name="hair"),
    #Label(index=2, color=(255, 229, 204), name="body skin"),
    #Label(index=3, color=(255, 204, 204), name="face skin"),
    Label(index=4, color=(0, 0, 128), name="clothes"),
    #Label(index=5, color=(128, 0, 128), name="others"),
]
SegmentationLabels = SegmentationMap(labels)


# helper for postprocessing output mask
def postprocess_mask(out: np.ndarray, pad_info: Tuple[int, int], orig_img_size: Tuple[int, int]):
    """
    Posptprocessing function for segmentation mask, accepts model output tensor,
    gets labels for each pixel using argmax,
    unpads segmentation mask and resizes it to original image size.

    Parameters:
      out (np.ndarray): model output tensor
      pad_info (Tuple[int, int]): information about padding size from preprocessing step
      orig_img_size (Tuple[int, int]): original image height and width for resizing
    Returns:
      label_mask_resized (np.ndarray): postprocessed segmentation label mask
    """
    label_mask = np.argmax(out, -1)[0]
    pad_h, pad_w = pad_info
    unpad_h = label_mask.shape[0] - pad_h
    unpad_w = label_mask.shape[1] - pad_w
    label_mask_unpadded = label_mask[:unpad_h, :unpad_w]
    orig_h, orig_w = orig_img_size
    label_mask_resized = cv2.resize(label_mask_unpadded, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return label_mask_resized


# Get info about original image
image_data = np.array(img)
orig_img_shape = image_data.shape

# Specify background color for replacement
BG_COLOR = (192, 192, 192)

# Blur image for backgraund blurring scenario using Gaussian Blur
blurred_image = cv2.GaussianBlur(image_data, (55, 55), 0)

# Postprocess output
postprocessed_mask = postprocess_mask(out, pad_info, orig_img_shape[:2])

# Get colored segmentation map
output_mask = segmentation_map_to_image(postprocessed_mask, SegmentationLabels.get_colormap())

# Replace background on original image
# fill image with solid background color
bg_image = np.full(orig_img_shape, BG_COLOR, dtype=np.uint8)

# define condition mask for separation background and foreground
condition = np.stack((postprocessed_mask,) * 3, axis=-1) > 0
# replace background with solid color
output_image = np.where(condition, image_data, bg_image)
# replace background with blurred image copy
output_blurred_image = np.where(condition, image_data, blurred_image)

import matplotlib.pyplot as plt

titles = ["Original image", "Portrait mask", "Removed background", "Blurred background"]
images = [image_data, output_mask, output_image, output_blurred_image]
figsize = (16, 16)
fig, axs = plt.subplots(2, 2, figsize=figsize, sharex="all", sharey="all")
fig.patch.set_facecolor("white")
list_axes = list(axs.flat)
for i, a in enumerate(list_axes):
    a.set_xticklabels([])
    a.set_yticklabels([])
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)
    a.grid(False)
    a.imshow(images[i].astype(np.uint8))
    a.set_title(titles[i])
fig.subplots_adjust(wspace=0.0, hspace=-0.8)
fig.tight_layout()

import collections
import time
from IPython import display
from typing import Union

from notebook_utils import VideoPlayer


# Main processing function to run background blurring
def run_background_blurring(
    source: Union[str, int] = 0,
    flip: bool = False,
    use_popup: bool = False,
    skip_first_frames: int = 0,
    model: ov.Model = ov_model,
    device: str = "CPU",
):
    """
    Function for running background blurring inference on video
    Parameters:
      source (Union[str, int], *optional*, 0): input video source, it can be path or link on video file or web camera id.
      flip (bool, *optional*, False): flip output video, used for front-camera video processing
      use_popup (bool, *optional*, False): use popup window for avoid flickering
      skip_first_frames (int, *optional*, 0): specified number of frames will be skipped in video processing
      model (ov.Model): OpenVINO model for inference
      device (str): inference device
    Returns:
      None
    """
    player = None
    compiled_model = core.compile_model(model, device)
    try:
        # Create a video player to play with target fps.
        player = VideoPlayer(source=source, flip=flip, fps=30, skip_first_frames=skip_first_frames)
        # Start capturing.
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        processing_times = collections.deque()
        while True:
            # Grab the frame.
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(
                    src=frame,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
            # Get the results.
            input_image, pad_info = resize_and_pad(frame, 256, 256)
            normalized_img = np.expand_dims(input_image.astype(np.float32) / 255, 0)

            start_time = time.time()
            # model expects RGB image, while video capturing in BGR
            segmentation_mask = compiled_model(normalized_img[:, :, :, ::-1])[0]
            stop_time = time.time()
            blurred_image = cv2.GaussianBlur(frame, (55, 55), 0)
            postprocessed_mask = postprocess_mask(segmentation_mask, pad_info, frame.shape[:2])
            condition = np.stack((postprocessed_mask,) * 3, axis=-1) > 0
            frame = np.where(condition, frame, blurred_image)
            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # Mean processing time [ms].
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(
                img=frame,
                text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=f_width / 1000,
                color=(255, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            # Use this workaround if there is flickering.
            if use_popup:
                cv2.imshow(winname=title, mat=frame)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
            else:
                # Encode numpy array to jpg.
                _, encoded_img = cv2.imencode(ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
                # Create an IPython image.
                i = display.Image(data=encoded_img)
                # Display the image in this notebook.
                display.clear_output(wait=True)
                display.display(i)
    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            # Stop capturing.
            player.stop()
        if use_popup:
            cv2.destroyAllWindows()

WEBCAM_INFERENCE = False

if WEBCAM_INFERENCE:
    VIDEO_SOURCE = 0  # Webcam
else:
    VIDEO_SOURCE = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/CEO%20Pat%20Gelsinger%20on%20Leading%20Intel.mp4"

run_background_blurring(source=VIDEO_SOURCE, device=device.value)