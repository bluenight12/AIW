import streamlit as st
import speech_recognition as sr
from streamlit_extras.switch_page_button import switch_page
from googletrans import Translator

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

            st.session_state["text"] = text
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
    f_input = st.session_state.get('text')
    korean_sentence = f_input.strip()
    english_translation = translator.translate(korean_sentence, src='ko', dest='en').text
    print(f"번역 결과: {english_translation}")
    st.session_state["trans_text"] = english_translation


def extract_clothes():
    """
    english.txt 파일에서 'yellow', 'black', 'blue', 't-shirt', 'cote'와 같은 단어를 찾아 
    clothes.txt 파일에 한 줄에 나열하여 저장하는 함수
    """
    # 'input_file_path' 파일 읽기
    english_file = st.session_state.get('trans_text')
    print(english_file)
    words = english_file.split()
    words = [i.lower() for i in words]
    # 'output_file_path' 파일에 발견된 단어들을 한 줄에 나열하여 저장
    st.session_state['extract_text'] = (' '.join([word for word in words if word in ['red', 'black', 'blue', '-shirt', 'green', 'man to man', 'knit', 'white', 'shirt','long sleeve']]) + '\n')

def navigate_previous():
    st.rerun()



def main():
    st.set_page_config(page_title="Streamlit WebCam App")  
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
    if st.button("이전 페이지로 이동"):
        navigate_previous()
    #st.markdown("<h2 style='text-align: center; color: white;'>버튼을 누르고 원하는 옷을 말해주세요</h2>", unsafe_allow_html=True)
    if 'text' not in st.session_state:
        st.session_state.text = ""
    if 'trans_text' not in st.session_state:
        st.session_state.trans_text = ""
    if 'extract_text' not in st.session_state:
        st.session_state.extract_text = ""
    frame_placeholder = st.empty()
    with frame_placeholder:
        with st.chat_message("ai"):
            st.markdown(f"<div style = text-align:center;>버튼을 누르고 원하는 옷을 말해주세요</div>", unsafe_allow_html=True)
    # frame_placeholder.image(st.session_state.get("image"), channels='RGB')
    
    cols = st.columns(6)
    with cols[1]:
        text_button_pressed = st.button("음성 받기", use_container_width=True)
        
    with cols[4]:
        next_page_button = st.button("넘어가기", use_container_width=True)
        
    if text_button_pressed:
        recognize_and_save_text()
        if st.session_state.text != "":
            with frame_placeholder:
                with st.chat_message("user"):
                    text = st.session_state.get('text')
                    st.markdown(f"<div style = text-align:center;>{text}</div>", unsafe_allow_html=True)
                translate_korean_to_english()
                extract_clothes()
                st.session_state["text"] = ""
        else:
            with frame_placeholder:
                with st.chat_message("ai"):
                    st.markdown(f"<div style = text-align:center;>다시 한 번 말해주세요</div>", unsafe_allow_html=True)

    if next_page_button:
        switch_page("make_image")
    
    if st.button("이전 페이지로 이동"):
        navigate_previous()
    
if __name__ == "__main__":
    main()
