import streamlit as st
import stream
import numpy as np
from PIL import Image

st.title('hello')
uploaded=st.file_uploader('Choose',type=['jpeg','png'])
if uploaded is not None:
    image=Image.open(uploaded)
    label=stream.predict(uploaded)
    st.write(label)