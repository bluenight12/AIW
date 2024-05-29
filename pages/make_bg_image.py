import os
import glob
import cv2
import streamlit as st
from refer import webui_api as api
from util import Clothes_Segmentation as cs
import time
import sqlite3
import numpy as np
import requests
from streamlit_extras.switch_page_button import switch_page
from imgur_python import Imgur

def Recent_file():
    files = glob.glob('api_out/img2img/img2img_*.png')
    recent_file = max(files, key=os.path.getmtime)
    return recent_file

def Make_img():
    #프롬프트 TXT파일에서 가져오는 코드 만들기 

    bg = st.session_state['bg_trans_text']
    
    ## 세션에서 프롬프트 바로 넣어주기
    prompt = st.session_state.get("bg_trans_text")
    # ap.t2i(prompt)
    input_img_path = ("Cam.jpg")
    ## U2net으로 bg 제거
    mask_img_path = ("mask.jpg")
    cs.Cloths_seg(input_img_path,mask_img_path)
    ap = api.Create_image()
    ap.i2i(input_img_path, mask_img_path, prompt)
    empty_space.progress(100)
    # bg_prompt = ("")
    # ap.t2i(bg_prompt)

def main():
    st.set_page_config(page_title="Streamlit WebCam App")
    st.markdown("""
        <div style='text-align: left;'>
            <a href="/" target="_self">
                <button style='background-color: #0068c9; color: white; padding: 10px 20px; border-radius: 5px; border: none; font-size: 16px; cursor: pointer; height: 50px'>처음으로</button>
            </a>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<h1 style = text-align:center;>QR코드를 스캔하시면<br>이미지를 가져가실 수 있어요 !</h1>", unsafe_allow_html=True)
    text = st.session_state.get("text")
    global empty_space
    
    st.markdown(
    """
    <style>
    button {
        height: 100px;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    imgur_client = Imgur({"client_id": "57caf94e3b6438c"})
    #image = imgur_client.image_upload(os.path.realpath('./image.png'), 'Untitled', 'My first image upload')
    #image_id = image['response']['data']['id']
    #print(image_id)
    cols = st.columns(2)
    place_holder = st.empty()
    with place_holder:
        with st.chat_message("ai"):
            st.markdown(f"<div style = text-align:center;></div>", unsafe_allow_html=True)
        
    frame_holder = st.empty()

    frame_holder.image(st.session_state.get("final_image"), channels="RGB", use_column_width=True)


if __name__ == "__main__":
    if "bg_trans_text" not in st.session_state:
        st.session_state.extract_text="hawaii beach"
    main()