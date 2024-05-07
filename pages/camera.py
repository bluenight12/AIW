import streamlit as st
import cv2
from streamlit_extras.switch_page_button import switch_page
from refer import age_gender as ag
import time

age_gender_bin_path = './models/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.bin'
age_gender_xml_path = './models/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml'
face_detection_bin_path = './models/face-detection-retail-0005/FP16/face-detection-retail-0005.bin'
face_detection_xml_path = './models/face-detection-retail-0005/FP16/face-detection-retail-0005.xml'

def btn_disable(flag):
    st.session_state['btn_disable'] = flag
    

def main():
    st.title("Webcam Display Steamlit App")
    st.caption("Powered by OpenCV, Streamlit")
    cap = cv2.VideoCapture(1)

    ### age_gender, face_detection 초기화 ###
    face_detector = ag.FaceDetection(model_bin=face_detection_bin_path, model_xml=face_detection_xml_path)
    age_gender_predictor = ag.AgeGenderPrediction(model_bin=age_gender_bin_path, model_xml=age_gender_xml_path)
    
    frame_placeholder = st.empty()
    col1, col2 = st.columns([1, 1])

    with col1:
        shutter_button_pressed = st.button("촬영", on_click=btn_disable, args=(False, ))  # 자동 촬영으로 바꾸기

    with col2:
        next_button_pressed = st.button("넘어가기", disabled=st.session_state.get("btn_disable"), on_click=btn_disable, args=(True, ))

    reset_button_pressed = st.button("다시 찍기")



    if next_button_pressed:
        switch_page("voice")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Video Capture Ended")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame, 1)
        frame_placeholder.image(frame, channels="RGB")
        if next_button_pressed:
            break
        elif shutter_button_pressed:
            cv2.imwrite("Cam.jpg", cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (480, 640)))
            st.session_state.image = frame
            faces = face_detector.run(frame)
            try:
                age, gender = age_gender_predictor.run(faces)
            
            except ValueError as e:
                raise ValueError("Image is empty") from e
            except cv2.error as e:
                print(e)
            
            st.session_state['age'] = age
            st.session_state['gender'] = gender
            break
        cv2.waitKey(33)



if __name__ == "__main__":
    if 'image' not in st.session_state:
        st.session_state.image = ""
    if 'age' not in st.session_state:
        st.session_state.age = ""
    if 'gender' not in st.session_state:
        st.session_state.gender = ""
    if 'btn_disable' not in st.session_state:
        st.session_state.btn_disable = True
    main()
