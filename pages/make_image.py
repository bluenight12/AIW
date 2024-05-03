import streamlit as st
from refer import reference

def main():
    st.set_page_config(page_title="Streamlit WebCam App")
    st.title("Image Test")
    text = st.session_state.get("text")
    make_button_pressed = st.button("이미지 만들기", on_click=reference.print_text, args=("Hi",))

if __name__ == "__main__":
    main()