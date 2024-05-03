import streamlit as st
import speech_recognition as sr

def recognize_and_save_text(output_file):
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
            text = recognizer.recognize_whisper(audio, language='ko-KR')
            print("음성 인식 결과:", text)

            # 인식된 텍스트를 text.txt 파일에 저장
            with open(output_file, "w") as file:
                file.write(text)
            print("인식된 단어를 text.txt 파일에 저장했습니다.")
            return text

        except sr.UnknownValueError:
            return list("음성을 인식할 수 없습니다. 다시 시도해주세요.")
        except sr.RequestError as e:
            return list("Google Speech Recognition 서비스에 접근할 수 없습니다; {0}".format(e))
        
def main():
    st.set_page_config(page_title="Streamlit WebCam App")
    st.title("Voice Test")
    text_button_pressed = st.button()
    if text_button_pressed:
        recognize_and_save_text("text.txt")
        text_button_pressed = False
    