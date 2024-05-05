import os
import glob
import cv2
import streamlit as st
from refer import webui_api as api
from util import Clothes_Segmentation as cs
def Recent_file():
    files = glob.glob('api_out/img2img/img2img-*.png')
    recent_file = max(files, key=os.path.getmtime)
    return recent_file
def Make_img():

    prompt = ("")
    input_img_path = ("Cam.jpg")
    mask_img_path = ("mask.jpg")
    cs.Cloths_seg(input_img_path,mask_img_path)
    ap = api.Create_image()
    ap.i2i(input_img_path, mask_img_path, prompt)
def main():
    st.set_page_config(page_title="Streamlit WebCam App")
    st.title("Image Test")
    text = st.session_state.get("text")
    frame_placeholder = st.empty()
    make_button_pressed = st.button("이미지 만들기",  args=("Hi",))



    if make_button_pressed:
        Make_img()
        image = cv2.imread(Recent_file())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(image, channels="RGB")
        pass


if __name__ == "__main__":
    main()