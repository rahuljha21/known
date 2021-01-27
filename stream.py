  
import streamlit as st
import tensorflow.keras as keras
import numpy
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions


@st.cache(allow_output_mutation=True)
def get_model():
    model=VGG16()
    return model

def predict(image):
    loaded=get_model()
    image=load_img(image,target_size=(224,224,3))
    image=img_to_array(image)
    image=image/255.0
    image=np.reshape(image,[1,224,224,3])
    image=preprocess_input(image)
    classes=loaded.predict(image)
    return classes
