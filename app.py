import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Image Sense ", page_icon='📸', layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Image Sense 🌆")

# Instruction to user about the model classes
st.info("""
This model recognizes **5 classes** only:
- Dew
- Forest
- Glacier
- Mountain
- Plastic Bottles

📌 Upload a clear image representing one of the above categories.
""")

st.markdown("""
### ℹ️ Model Info:
This image classifier supports only **5 classes**:
- 🌱 Dew  
- 🌳 Forest  
- ❄️ Glacier  
- 🏔️ Mountain  
- 🧴 Plastic Bottles  

Please upload an image representing one of the above for accurate prediction.
""")


st.code("""
Supported Classes:
1. Dew
2. Forest
3. Glacier
4. Mountain
5. Plastic Bottles
""", language='markdown')


st.caption("Note: This model only works with Dew, Forest, Glacier, Mountain, and Plastic Bottles.")


uploaded_file = st.file_uploader(label="Upload your image")

model = load_model('image_reco_vgg.keras')

def get_img_input(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

if uploaded_file:
    image = Image.open(uploaded_file)
    image.thumbnail((300, 300))  # Resize for consistent display
    st.image(image, caption='Uploaded Image', use_container_width=False)

    button = st.button("Classify")
    if button:
        img = get_img_input(uploaded_file)
        result = model.predict(img)
        y_pred2 = pd.DataFrame(np.argmax(result, axis=1))
        print(y_pred2)

        y_pred = y_pred2[0][0]

        if y_pred == 0:
            st.subheader("Predicted Class: Dew")
        elif y_pred == 1:
            st.subheader("Predicted Class: Forest")
        elif y_pred == 2:
            st.subheader("Predicted Class: Glacier")
        elif y_pred == 3:
            st.subheader("Predicted Class: Mountain")
        elif y_pred == 4:
            st.subheader("Predicted Class: Plastic Bottles")
