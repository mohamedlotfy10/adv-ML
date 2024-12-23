import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("satellite_cnn2.h5")

# Class labels
class_labels = ["Annual Crop", "Forest", "Herbaceous Vegetation", "Highway", 
                "Industrial", "Pasture", "Permanent Crop", "Residential", 
                "River", "Sea/Lake"]

# Function to predict land type
def predict_land_type(image):
    image = image.resize((64, 64))  # Resize to match model input
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)[0]
    class_index = np.argmax(prediction)
    confidence = prediction[class_index]
    return class_labels[class_index], confidence

# Streamlit App
st.title("Satellite Image Classification")
st.write("Upload a satellite image to classify it into a land-use category.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    
    # Predict
    label, confidence = predict_land_type(image)
    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: {confidence:.2f}")