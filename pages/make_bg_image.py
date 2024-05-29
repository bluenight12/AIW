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

    st.markdown(f"<h1 style = text-align:center;>버튼을 누르고 기다려주세요</h1>", unsafe_allow_html=True)
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
    
    cols = st.columns(2)
    place_holder = st.empty()
    with place_holder:
        with st.chat_message("ai"):
            st.markdown(f"<div style = text-align:center;></div>", unsafe_allow_html=True)
        
    frame_holder = st.empty()

    empty_space =  st.progress(0)
    btn_cols = st.columns(6)
    with btn_cols[1]:
        make_button_pressed = st.button("이미지\n\n 만들기", key="make_button", use_container_width=True)
    
    with btn_cols[4]:
        if st.button("결과 보기"):
            switch_page("edit_page")
            st.rerun()
        
    if make_button_pressed:
        Make_img()
        with place_holder:
            with st.chat_message("ai"):
                st.markdown(f"<div style = text-align:center;>다 만들었으니 결과로 넘어가주세요!</div>", unsafe_allow_html=True)
        # image = cv2.imread(Recent_file())
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # frame_placeholder.image(image, channels="RGB")
        pass


if __name__ == "__main__":
    if "bg_trans_text" not in st.session_state:
        st.session_state.extract_text="hawaii beach"
    main()