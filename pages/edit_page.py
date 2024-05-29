import streamlit as st
import glob
from natsort import natsorted
from streamlit_extras.switch_page_button import switch_page
import qrcode
import cv2

def main():
    st.markdown("""
            <div style='text-align: left;'>
                <a href="/" target="_self">
                    <button style='background-color: #0068c9; color: white; padding: 10px 20px; border-radius: 5px; border: none; font-size: 16px; cursor: pointer; height: 50px;'>처음으로</button>
                </a>
            </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div style='display: flex; align-items: center; justify-content: center;'>
            <div style='text-align: center;'>
                <h1>합성 결과입니다</h1>
            </div>
        </div>
    """, unsafe_allow_html=True)
    files = glob.glob('api_out/img2img/img2img-0-*.png')
    recent_file = natsorted(seq=files, reverse=True)[0]
    print(recent_file)
    image_path = recent_file
    st.markdown(
        """
    <style>
    .element-container:nth-of-type(1) button {
        height: 100px;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
        background-color: #baed91;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    image = st.image(image_path, use_column_width=True)
    
    link = st.session_state.get("link")
    print(link)
    myqr = qrcode.make(link)
    myqr.save(stream='qr_code.png')
    img = cv2.imread('qr_code.png')
    i = cv2.resize(img, dsize=(140, 140), interpolation=cv2.INTER_LANCZOS4)
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.markdown(f"<div style = text-align:center;justify-content:center;font-size:40px; >옷 정보 링크</div>", unsafe_allow_html=True)
    with col2:
        st.image(i)
    with col3:
        bg_button = st.button("배경 합성하기")
    
    if bg_button:
        switch_page("bg_voice")
        st.rerun()
    
if __name__ == "__main__":
    main()