import streamlit as st
import speech_recognition as sr
from streamlit_extras.switch_page_button import switch_page
from googletrans import Translator
from util import bg_Segmentation as bs
from refer import webui_api as api
# from models.change_background import ChangeBackground as cb
import glob
from natsort import natsorted
import cv2

def recognize_and_save_text():
    # 음성 인식기 객체 생성
    recognizer = sr.Recognizer()

    while True:
        # 마이크로 음성 입력 받기
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0)  # 배경 소음에 대한 자동 조절
            print("말씀해주세요...")
            try:
                audio = recognizer.listen(source, timeout=1.5, phrase_time_limit=3)  # 마이크에서 음성 입력을 듣기
            except sr.WaitTimeoutError:
                return list("음성이 입력되지 않았습니다.")

        # 음성을 텍스트로 변환
        try:
            text = recognizer.recognize_google(audio, language='ko-KR')
            #text = recognizer.recognize_wit(audio, key='QC3S7TF4MGTOXKSBLAX32VPK77RV5WDV')
            #text = recognizer.recognize_whisper(audio, language='ko')
            
            print("음성 인식 결과:", text)

            st.session_state["bg_text"] = text
            return

        except sr.UnknownValueError:
            return list("음성을 인식할 수 없습니다. 다시 시도해주세요.")
        except sr.RequestError as e:
            return list("Google Speech Recognition 서비스에 접근할 수 없습니다; {0}".format(e))
        
def translate_korean_to_english():
    """
    한국어 문장을 영어로 번역하여 결과를 파일에 저장합니다.

    Args:
    input_file (str): 한국어 문장이 있는 파일 경로.
    output_file (str): 번역된 영어 문장을 저장할 파일 경로.
    """
    # Google 번역기 객체 생성
    translator = Translator()

    # 한국어 문장을 읽어와서 영어로 번역하여 저장


    # 한국어 문장을 읽어와서 영어로 번역하여 저장
    f_input = st.session_state.get('bg_text')
    korean_sentence = f_input.strip()
    # translation = translator.translate(korean_sentence)
    english_translation = translator.translate(korean_sentence, src='ko', dest='en').text
    print(f"번역 결과: {english_translation}")
    st.session_state["bg_trans_text"] = english_translation


def navigate_previous():
    switch_page("edit_page")
    st.rerun()

def make_img(holder, bar):
    # t2i, st.session_state의 bg_trans_text가 영문으로 된 음성 세션입니다.

    image_path = st.session_state.get('cloth_gen_image_path')
    bg_mask_path = 'img/bg_mask.png'
    prompt = st.session_state.get('bg_trans_text')
    bg_prompt = f"background : {prompt}"
    bs.bg_seg(image_path, bg_mask_path)
    ap = api.Create_image()
    ap.i2i(image_path, bg_mask_path, bg_prompt)

    print("이미지 생성 후 저장")
    bar.progress(50)
    #holder 는 테스트용 Cam.jpg 불러오는 코드입니다.
    bg_gen_file = glob.glob('api_out/img2img/img2img-0-*.png')
    bg_gen_recent_file = natsorted(seq=bg_gen_file, reverse=True)[0]
    bg_gen_image_path = bg_gen_recent_file
    test = cv2.cvtColor(cv2.imread(bg_gen_image_path), cv2.COLOR_BGR2RGB)
    holder.image(test, channels="RGB", use_column_width=True)
    st.session_state.final_image = test
    bar.progress(100)

    # #####
    # prompt = st.session_state.get('bg_trans_text')
    # bg_gen = api.Create_image()
    # bg_gen.t2i(prompt)
    # bar.progress(50)
    # background_files = glob.glob('api_out/txt2img/txt2img-*.png')
    # recent_file = natsorted(seq=background_files, reverse=True)[0]
    # background_image_path = recent_file
    # test = cv2.cvtColor(cv2.imread(background_image_path), cv2.COLOR_BGR2RGB)
    #
    # background_changer = cb(image_path, background_image_path, "final_image.png")
    # holder.image(background_changer, channels="RGB", use_column_width=True)
    # bar.progress(100)





def main():
    st.set_page_config(page_title="Streamlit WebCam App") 
    st.markdown("""
        <div style='text-align: left;'>
            <a href="/edit_page" target="_self">
                <button style='background-color: #0068c9; color: white; padding: 10px 20px; border-radius: 5px; border: none; font-size: 16px; cursor: pointer; height: 50px'>이전으로</button>
            </a>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(
        """
    <style>
    button {
        height: auto;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    # if st.button("이전 페이지로 이동"):
    #     navigate_previous()
    #st.markdown("<h2 style='text-align: center; color: white;'>버튼을 누르고 원하는 옷을 말해주세요</h2>", unsafe_allow_html=True)
    if 'text' not in st.session_state:
        st.session_state.text = ""
    if 'trans_text' not in st.session_state:
        st.session_state.trans_text = ""
    if 'extract_text' not in st.session_state:
        st.session_state.extract_text = ""
    ###
    ### 배경을 여러번 바꿀시 기존이미지 경로가 최신경로가 아니게 됨으로 여기에서 경로를 임시로 지정했습니다.
    cloth_generation_files = glob.glob('api_out/img2img/img2img-0-*.png')
    recent_file = natsorted(seq=cloth_generation_files, reverse=True)[0]
    cloth_generation_image_path = recent_file
    st.session_state.cloth_gen_image_path = cloth_generation_image_path
    ###
    frame_placeholder = st.empty()
    with frame_placeholder:
        with st.chat_message("ai"):
            st.markdown(f"<div style = font-size:30px;text-align:center;>버튼을 누르고 원하는 배경을 말해주세요</div>", unsafe_allow_html=True)
    # frame_placeholder.image(st.session_state.get("image"), channels='RGB')
    img_placeholder = st.empty()
    progressBar =  st.progress(0)
    cols = st.columns(7)
    with cols[1]:
        text_button_pressed = st.button("음성받기", use_container_width=True)
    with cols[3]:
        make_button_pressed = st.button("합성하기", use_container_width=True)
    with cols[5]:
        next_page_button = st.button("넘어가기", use_container_width=True)
        
    if text_button_pressed:
        recognize_and_save_text()
        if st.session_state.bg_text != "":
            with frame_placeholder:
                with st.chat_message("user"):
                    text = st.session_state.get('bg_text')
                    st.markdown(f"<div style = font-size:30px;text-align:center;>{text}</div>", unsafe_allow_html=True)
                translate_korean_to_english()
                st.session_state["bg_text"] = ""
        else:
            with frame_placeholder:
                with st.chat_message("ai"):
                    st.markdown(f"<div style = font-size:30px;text-align:center;>다시 한 번 말해주세요</div>", unsafe_allow_html=True)

    if make_button_pressed:
        make_img(img_placeholder, progressBar)
        with frame_placeholder:
            with st.chat_message("ai"):
                st.markdown(f"<div style =  font-size:30px;text-align:center;> 이쁘게 합성되었어요 !</div>", unsafe_allow_html=True)

    if next_page_button:
        switch_page("make_bg_image")

    
if __name__ == "__main__":
    if "bg_text" not in st.session_state:
        st.session_state.bg_text = ""
    if "bg_trans_text" not in st.session_state:
        st.session_state.bg_trans_text = ""
    if "cloth_gen_image_path" not in st.session_state:
        st.session_state.cloth_gen_image_path = ""
    if "final_image" not in st.session_state:
        st.session_state.final_image = ""
    main()
