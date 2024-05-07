def Background_removal():
    import os
    import cv2
    import requests
    import numpy as np
    import openvino as ov
    import ipywidgets as widgets
    from typing import Tuple
    from pathlib import Path
    from notebook_utils import load_image
    from notebook_utils import download_file
    from notebook_utils import segmentation_map_to_image, SegmentationMap, Label

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)

    tflite_model_path = Path("selfie_multiclass_256x256.tflite")
    tflite_model_url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
    download_file(tflite_model_url, tflite_model_path)
    core = ov.Core()
    ir_model_path = tflite_model_path.with_suffix(".xml")
    if not ir_model_path.exists():
        ov_model = ov.convert_model(tflite_model_path)
        ov.save_model(ov_model, ir_model_path)
    else:
        ov_model = core.read_model(ir_model_path)
    print(f"Model input info: {ov_model.inputs}")
    print(f"Model output info: {ov_model.outputs}")

    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="AUTO",
        description="Device:",
        disabled=False,
    )
    device

    compiled_model = core.compile_model(ov_model, device.value)

    # Read input image and convert it to RGB
    test_image_url = "C:\\Users\\admin\\Documents\\test_git\\AIW\\img\\test4.jpg"
    img = load_image(test_image_url)
    
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

    # helper for visualization segmentation labels
    labels = [
        Label(index=0, color=(192, 192, 192), name="background"),
        Label(index=1, color=(128, 0, 0), name="hair"),
        Label(index=2, color=(255, 229, 204), name="body skin"),
        Label(index=3, color=(255, 204, 204), name="face skin"),
        Label(index=4, color=(0, 0, 128), name="clothes"),
        Label(index=5, color=(128, 0, 128), name="others"),
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

    titles = ["Original image", "Portrait mask", "Removed background", "Blurred background"]
    images = [image_data, output_mask, output_image, output_blurred_image]
 
    # Get the index of the "Removed background" image
    removed_background_index = titles.index("Removed background")
    # Get the "Removed background" image
    removed_background_image = images[removed_background_index]
    # Define the path to save the "Removed background" image
    output_removed_background_path = "C:\\Users\\admin\\Documents\\test_git\\AIW\\img\\removed_background.jpg"
    # Save the "Removed background" image
    cv2.imwrite(output_removed_background_path, removed_background_image)


Background_removal()