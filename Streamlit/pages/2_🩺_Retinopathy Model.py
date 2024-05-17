import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title= "Retinopathy",
    page_icon = "👁️"
)

st.sidebar.success("Select a Page")

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('Streamlit/model3 (1).h5')
  return model
model=load_model()
st.write("""
# Diabetic Retinopathy Detection System"""
)
file=st.file_uploader("Choose retina image from computer",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(150,150)
    image=ImageOps.fit(image_data,size,Image.Resampling.LANCZOS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file).convert("RGB")
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['No Diabetic Retinopathy', 'Signs of Diabetic Retinopathy']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import scipy as sc

# def import_and_predict(image, model):
#     new_shape = (150,150,3)
#     X_data_resized = [sc.misc.imresize(image, new_shape) for image in X_data]
#     img = np.asarray(X_data_resized)
#     img_reshape = img[np.newaxis,...]
#     prediction = model.predict(img_reshape)
#     return prediction
# if file is None:
#     st.text("Please upload an image file")
# else:
#     image = Image.open(file)
#     st.image(image,use_column_width=True)
#     prediction = import_and_predict(image,model)
#     class_names=['No Diabetic Retinopathy', 'Signs of Diabetic Retinopathy']
#     string="OUTPUT : "+class_names[np.argmax(prediction)]
#     st.success(string)
