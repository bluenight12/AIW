import os
import glob
import cv2
import streamlit as st
from refer import webui_api as api
from util import Clothes_Segmentation as cs
import time
import sqlite3

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
    
def get_cloth():
    con = sqlite3.connect('./db/cloth.db')
    cur = con.cursor()
    clothes = st.session_state['extract_text']
    words = clothes.split()
    print(words)
    ctg = [word for word in words if word in [
        '-shirt', 'long sleeve', 'man to man', 'knit']]
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
        f'SELECT id, Image_Link FROM cloth WHERE category="{ctg}" AND color="{color}" ORDER BY RANDOM() LIMIT 3;')
    cloth_list = cur.fetchall()

    

def main():
    st.set_page_config(page_title="Streamlit WebCam App")
    st.title("Image Test")
    text = st.session_state.get("text")
    frame_placeholder = st.empty()
    make_button_pressed = st.button("이미지 만들기")
    
    cols = st.columns(2)
    container1 = cols[0].container(height=400)
    container2 = cols[1].container(height=400)
    container1.write('Meow' + ' meow'*1000)
    container2.write('Meow' + ' meow'*1000)

    if make_button_pressed:
        Make_img()
        image = cv2.imread(Recent_file())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(image, channels="RGB")
        pass


if __name__ == "__main__":
    main()