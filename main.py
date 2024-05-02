import streamlit as st

def main():
    # 텍스트와 버튼 수직 중앙 정렬
    st.markdown("""
        <div style='display: flex; align-items: center; justify-content: center; height: 80vh;'>
            <div style='text-align: center;'>
                <h1>안녕하세요!</h1>
                <h1></h1><h1></h1><h1></h1>
                <a href="/camera" target="_self">
                    <button style='background-color: #0068c9; color: white; padding: 10px 20px; border-radius: 5px; border: none; font-size: 16px; cursor: pointer;'>시작하기</button>
                </a>
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()