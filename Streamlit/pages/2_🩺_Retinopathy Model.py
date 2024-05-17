import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image,ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

st.set_page_config(
    page_title= "Retinopathy",
    page_icon = "👁️"
)

st.sidebar.success("Select a Page")

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('Streamlit/pages/final_model.h5')
  return model
model=load_model()
st.write("""
# Diabetic Retinopathy Detection System"""
)
file = st.file_uploader("Choose retina image from computer",type=["jpg","png","jpeg"])


def resize_img(file):
    img = cv2.imread(file)
    if img is None:
        return None, None
    resized = cv2.resize(img, (150, 150))
    expanded = resized[np.newaxis, ...]
    return resized, expanded

def import_and_predict(image_data,model):
    img, expanded_img = resize_img(image)
    prediction = model.predict(expanded_img)[0]
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    #image=Image.open(file).convert("RGB")
    image = load_img(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names = ['No Diabetic Retinopathy', 'Signs of Diabetic Retinopathy']
    final_prediction = np.argmax(prediction)
    string = "OUTPUT: " + str(prediction)
    st.success(string)
    # if final_prediction > 0.3:
    #     string = "OUTPUT: " + class_names[1]
    #     st.success(string)
    # else:
    #     string = "OUTPUT: " + class_names[0]
    #     st.success(string)
        
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
