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

def Recent_file():
    files = glob.glob('api_out/img2img/img2img-*.png')
    recent_file = max(files, key=os.path.getmtime)
    return recent_file
def Make_img():
    #프롬프트 TXT파일에서 가져오는 코드 만들기 
    con = sqlite3.connect('./db/cloth.db')
    cur = con.cursor()
    clothes = st.session_state['extract_text']
    words = clothes.split()
    print(words)
    ctg = [word for word in words if word in [
        '-shirt', 'long sleeve', 'man to man', 'knit', 'shirts']]
    color = [word for word in words if word in [
        'red', 'green', 'black', 'blue', 'white']]
    if ctg == [] or color == []:
        # 돌아가기 누르라고 해야함
        st.title("돌아가기를 누르고 다시 입력해주세요")
    ctg = ctg[-1]
    color = color[-1]
    if ctg == '-shirt':
        ctg = 't-shirt'
    cur.execute(
        f'SELECT lora FROM clothes WHERE category="{ctg}" AND color="{color}" ORDER BY RANDOM() LIMIT 1;')
    lora = cur.fetchall()
    print(lora)
    prompt = ("")
    with open(lora[0][0], 'r') as file:
        data = file.read()
        start_index = data.find('prompt :') + len('prompt :')
        end_index = data.find('Negative prompt:')
        result = data[start_index:end_index].strip()
    empty_space.progress(50)
    print(result)
    prompt = (result)
    input_img_path = ("Cam.jpg")
    mask_img_path = ("mask.jpg")
    cs.Cloths_seg(input_img_path,mask_img_path)
    ap = api.Create_image()
    ap.i2i(input_img_path, mask_img_path, prompt)
    empty_space.progress(100)
    # bg_prompt = ("")
    # ap.t2i(bg_prompt)
    
def get_color_cloth():
    con = sqlite3.connect('./db/cloth_original.db')
    cur = con.cursor()
    clothes = st.session_state['extract_text']
    words = clothes.split()
    print(words)
    ctg = [word for word in words if word in [
        '-shirt', 'long sleeve', 'man to man', 'knit', 'shirts']]
    color = [word for word in words if word in [
        'red', 'green', 'black', 'blue', 'white']]
    if ctg == [] or color == []:
        # 돌아가기 누르라고 해야함
        st.title("돌아가기를 누르고 다시 입력해주세요")
    ctg = ctg[-1]
    color = color[-1]
    if ctg == '-shirt':
        ctg = 't-shirt'
    cur.execute(
        f'SELECT id, Image_Link FROM cloth WHERE category="{ctg}" AND color="{color}" ORDER BY RANDOM() LIMIT 5;')
    cloth_list = cur.fetchall()
    return cloth_list

def get_age_cloth():
    con = sqlite3.connect('./db/cloth_original.db')
    cur = con.cursor()

    cur.execute(
        f'SELECT id, Image_Link FROM cloth WHERE age="{round(int(st.session_state.get("age")), -1)}" AND gender="{st.session_state.get("gender")}" ORDER BY RANDOM() LIMIT 5;')
    cloth_list = cur.fetchall()
    return cloth_list

def get_normal_image(url):
    image_nparray = np.asarray(
        bytearray(requests.get(url).content), dtype=np.uint8)
    image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
    return image

def main():
    st.set_page_config(page_title="Streamlit WebCam App")
    st.title("Image Test")
    text = st.session_state.get("text")
    global empty_space
    empty_space =  st.progress(0)
    frame_placeholder = st.empty()
    make_button_pressed = st.button("이미지 만들기")
    
    cols = st.columns(2)
    
    color_list = get_color_cloth()
    color_cloth_list = []
    
    for j in range(len(color_list)):
        i = get_normal_image(color_list[j][1])
        i = cv2.resize(i, (300, 400), interpolation=cv2.INTER_LANCZOS4)
        color_cloth_list.append(i)
        
    age_list = get_age_cloth()
    age_cloth_list = []
    
    for j in range(len(age_list)):
        i = get_normal_image(age_list[j][1])
        i = cv2.resize(i, (300, 400), interpolation=cv2.INTER_LANCZOS4)
        age_cloth_list.append(i)

    container1 = cols[0].container(height=400)
    container2 = cols[1].container(height=400)
    container1.image(color_cloth_list, channels='BGR')
    container2.image(age_cloth_list, channels='BGR')

    if make_button_pressed:
        Make_img()
        image = cv2.imread(Recent_file())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(image, channels="RGB")
        pass


if __name__ == "__main__":
    if "extract_text" not in st.session_state:
        st.session_state.extract_text="black t -shirt"
    if "age" not in st.session_state:        
        st.session_state.age='20'
    if "gender" not in st.session_state:        
        st.session_state.gender='male'
    main()