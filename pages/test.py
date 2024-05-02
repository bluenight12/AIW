from pyparsing import empty
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(layout="wide")
empty1,con1,empty2 = st.columns([0.3,1.0,0.3])
empyt1,con2,con3,empty2 = st.columns([0.3,0.5,0.5,0.3])
empyt1,con4,empty2 = st.columns([0.3,1.0,0.3])
empyt1,con5,con6,empty2 = st.columns([0.3,0.5,0.5,0.3])

with empty1 :
    empty() # 여백부분1

with con1 :
    st.markdown("<h1 style='text-align: center; color: grey;'>Big headline</h1>", unsafe_allow_html=True)

with con2 :
    st.text("bb")
    st.text("bb")
    st.text("bb")

with con3 :
    st.text("cc")
    
with con4 :
    st.text("dd")

with con5 :
    st.text("ee") 

with con6 :
    st.text("ff")
    st.text("ff")

with empty2 :
    empty() # 여백부분2