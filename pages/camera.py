import streamlit as st
import cv2
from streamlit_extras.switch_page_button import switch_page
from refer import age_gender as ag
import time
import asyncio
import mediapipe as mp
import math

age_gender_bin_path = './models/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.bin'
age_gender_xml_path = './models/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml'
face_detection_bin_path = './models/face-detection-retail-0005/FP16/face-detection-retail-0005.bin'
face_detection_xml_path = './models/face-detection-retail-0005/FP16/face-detection-retail-0005.xml'

def btn_disable(flag):
    st.session_state['btn_disable'] = flag
    
def shutter_func():
    #btn_disable(False)
    #asyncio.run(countdown())
    print("d")
    
async def countdown():
    count = 4
    for i in range(5):
        await asyncio.sleep(1)
        with countdown_holder:
            st.markdown(f"<h1 style = text-align:center;>{count}</h1>", unsafe_allow_html=True)
        
        print(count)
        count -= 1

def main():
    st.markdown(f"<h1 style = text-align:center;>촬영을 누르고 자세를 잡아주세요</h1>", unsafe_allow_html=True)
    #st.caption("Powered by OpenCV, Streamlit")
    cap = cv2.VideoCapture(cv2.CAP_V4L2)
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    ### age_gender, face_detection 초기화 ##
    face_detector = ag.FaceDetection(model_bin=face_detection_bin_path, model_xml=face_detection_xml_path)
    age_gender_predictor = ag.AgeGenderPrediction(model_bin=age_gender_bin_path, model_xml=age_gender_xml_path)
    global count
    count = ""
    global countdown_holder
    countdown_holder = st.empty()
    with countdown_holder:
        st.markdown(f"<h1 style = text-align:center;>{count}</h1>", unsafe_allow_html=True)
    frame_placeholder = st.empty()
    cols= st.columns(5)
    flag = False
    photo_time = 0
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

    with cols[0]:
        shutter_button_pressed = st.button("촬영", on_click=shutter_func, use_container_width=True)  # 자동 촬영으로 바꾸기
        #shutter_button_pressed = st.button("촬영", on_click=btn_disable, args=(False, ), use_container_width=True)  # 자동 촬영으로 바꾸기

    with cols[2]:
        reset_button_pressed = st.button("다시 찍기", use_container_width=True)

    with cols[4]:
        next_button_pressed = st.button("넘어가기", disabled=st.session_state.get("btn_disable"), on_click=btn_disable, args=(True, ), use_container_width=True)

    if next_button_pressed:
        switch_page("voice")

    if st.session_state.get("btn_disable") is False:
        frame_placeholder.image(st.session_state.get("image"), channels="RGB", use_column_width=True)
        st.session_state.btn_disable = True
    else:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                st.write("Video Capture Ended")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame)
            if results.pose_landmarks:
                right_hand = (int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_PINKY].x * frame.shape[1]),
                                    int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_PINKY].y * frame.shape[0]))
                left_hand = (int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_PINKY].x * frame.shape[1]),
                                int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_PINKY].y * frame.shape[0]))
                
                right_elbow = int(
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y * frame.shape[0])
                left_elbow = int(
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y * frame.shape[0])

                left = 50
                right = 430
                print(left_hand[1], right_hand[1])
                if 0 < right_hand[1] < left and right < left_hand[1] < 480:
                    flag = True
                else:
                    flag = False

            frame = cv2.transpose(frame)
            frame = cv2.flip(frame, 1)
            frame_placeholder.image(frame, channels="RGB", use_column_width=True)

            if flag is True:
                if photo_time == 0:
                    photo_time = time.time()
                elif time.time() - photo_time > 5:
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
                    with countdown_holder:
                        st.markdown(f"<h1 style = text-align:center;>찰칵</h1>", unsafe_allow_html=True)
                    btn_disable(False)
                    st.rerun()
                    break
                elif time.time() - photo_time >= 0:
                    with countdown_holder:
                        st.markdown(f"<h1 style = text-align:center;>{math.ceil(5 - (time.time() - photo_time))}</h1>", unsafe_allow_html=True)
            else:
                with countdown_holder:
                    st.markdown(f"<h1 style = text-align:center;>{''}</h1>", unsafe_allow_html=True)
                photo_time = 0

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
            cv2.waitKey(1)
    cap.release()



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
