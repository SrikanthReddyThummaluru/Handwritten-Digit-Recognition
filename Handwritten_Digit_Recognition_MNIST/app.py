
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps

st.title("Handwritten Digit Recognition")
st.write("Upload an image of a digit (0-9)")

uploaded_file = st.file_uploader("Choose an image...", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image = ImageOps.invert(image)
    img = image.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    model = tf.keras.models.load_model("model.h5")
    prediction = model.predict(img_array)
    st.write(f"Prediction: {np.argmax(prediction)}")
    st.image(image, caption='Uploaded Image', use_column_width=True)
