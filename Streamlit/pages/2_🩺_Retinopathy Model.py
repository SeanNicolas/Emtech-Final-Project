import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title= "Retinopathy",
    page_icon = "👁️"
)

st.title("Main Page")
st.sidebar.success("Select a Page")

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('model3(1).hdf5')
  return model
model=load_model()
st.write("""
# Plant Leaf Detection System"""
)
file=st.file_uploader("Choose retina imageg from computer",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(64,64)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['No Diabetic Retinopathy', 'Signs of Diabetic Retinopathy']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
