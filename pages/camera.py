import streamlit as st
import cv2
from streamlit_extras.switch_page_button import switch_page
from refer import age_gender as ag

age_gender_bin_path = './model/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.bin'
age_gender_xml_path = './model/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml'
face_detection_bin_path = './model/face-detection-retail-0005/FP16/face-detection-retail-0005.bin'
face_detection_xml_path = './model/face-detection-retail-0005/FP16/face-detection-retail-0005.xml'

def main():
    st.set_page_config(page_title="Streamlit WebCam App")
    st.title("Webcam Display Steamlit App")
    st.caption("Powered by OpenCV, Streamlit@@@@@")
    cap = cv2.VideoCapture(1)

    ### age_gender, face_detection 초기화 ###
    face_detector = ag.FaceDetection(model_bin=face_detection_bin_path, model_xml=face_detection_xml_path)
    age_gender_predictor = ag.AgeGenderPrediction(model_bin=age_gender_bin_path, model_xml=age_gender_xml_path)
    
    frame_placeholder = st.empty()
    col1, col2 = st.columns([1, 1])
    with col1:
        start_button_pressed = st.button("시작")
    with col2:
        next_button_pressed = st.button("넘어가기")
    shutter_button_pressed = st.button("촬영")  # 자동 촬영으로 바꾸기
    if start_button_pressed:
        pass
    if next_button_pressed:
        switch_page("voice")
    while cap.isOpened() and not next_button_pressed:
        ret, frame = cap.read()
        if not ret:
            st.write("Video Capture Ended")
            break
        faces = face_detector.run(frame)
        try:
            age, gender = age_gender_predictor.run(faces)
        
        except ValueError as e:
            raise ValueError("Image is empty") from e
        except cv2.error as e:
            print(e)
            
        print(age, gender)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")
        if cv2.waitKey(1):
            if next_button_pressed:
                break
            elif shutter_button_pressed:
                cv2.imwrite("Cam.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                st.session_state.image = frame
                shutter_button_pressed = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if 'image' not in st.session_state:
        st.session_state.image = ""
    main()
