import streamlit as st
import speech_recognition as sr
from streamlit_extras.switch_page_button import switch_page

def recognize_and_save_text(output_file):
    # 음성 인식기 객체 생성
    recognizer = sr.Recognizer()

    while True:
        # 마이크로 음성 입력 받기
        kr_audio = sr.AudioFile('wav/news.wav')
        with kr_audio as source:
            audio = recognizer.record(source)
        # with sr.Microphone() as source:
        #     recognizer.adjust_for_ambient_noise(source, duration=0)  # 배경 소음에 대한 자동 조절
        #     print("말씀해주세요...")
        #     try:
        #         audio = recognizer.listen(source, timeout=1.5, phrase_time_limit=3)  # 마이크에서 음성 입력을 듣기
        #     except sr.WaitTimeoutError:
        #         return list("음성이 입력되지 않았습니다.")

        # 음성을 텍스트로 변환
        try:
            text = recognizer.recognize_wit(audio, key='QC3S7TF4MGTOXKSBLAX32VPK77RV5WDV')
            #text = recognizer.recognize_whisper(audio, language='ko')
            
            print("음성 인식 결과:", text)

            # 인식된 텍스트를 text.txt 파일에 저장
            with open(output_file, "w") as file:
                file.write(text)
            print("인식된 단어를 text.txt 파일에 저장했습니다.")
            st.session_state["text"] = text
            return

        except sr.UnknownValueError:
            return list("음성을 인식할 수 없습니다. 다시 시도해주세요.")
        except sr.RequestError as e:
            return list("Google Speech Recognition 서비스에 접근할 수 없습니다; {0}".format(e))
        
def main():
    st.set_page_config(page_title="Streamlit WebCam App")
    st.title("Voice Test")
    if 'text' not in st.session_state:
        st.session_state.text = ""
    frame_placeholder = st.empty()
    frame_placeholder.image(st.session_state.get("image"), channels='RGB')
    text_button_pressed = st.button("음성 받기")
    st.button("session state", on_click=(lambda a:print(a))(st.session_state['text']))
    
    if text_button_pressed:
        recognize_and_save_text("text.txt")
        print("exec")
        text_button_pressed = False
    
    next_page_button = st.button("넘어가기")
    if next_page_button:
        switch_page("make_image")
        next_page_button = False
    
if __name__ == "__main__":
    main()
