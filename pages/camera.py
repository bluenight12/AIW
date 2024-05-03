import streamlit as st
import cv2


def main():
    st.set_page_config(page_title="Streamlit WebCam App")
    st.title("Webcam Display Steamlit App")
    st.caption("Powered by OpenCV, Streamlit@@@@@")
    cap = cv2.VideoCapture(0)

    frame_placeholder = st.empty()
    col1, col2 = st.columns([1, 1])
    with col1:
        start_button_pressed = st.button("시작")
    with col2:
        next_button_pressed = st.button("넘어가기")
    shutter_button_pressed = st.link_button("촬영", "/pages/voice.py")  # 자동 촬영으로 바꾸기
    if start_button_pressed:
        pass
    while cap.isOpened() and not next_button_pressed:
        ret, frame = cap.read()
        if not ret:
            st.write("Video Capture Ended")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")
        if cv2.waitKey(1):
            if next_button_pressed:
                break
            elif shutter_button_pressed:
                cv2.imwrite("Cam.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                shutter_button_pressed = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
