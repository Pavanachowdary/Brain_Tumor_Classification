import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
import gdown
from PIL import Image

# Google Drive file ID for your model
file_id = "1EutGPLcz9fVKJKeZUMdVWl9PemQa1MAA"
model_url = f"https://drive.google.com/uc?id={file_id}"
model_path = "resnet50_model_4cls.h5"

# Function to download model if not present
def download_model():
    if not os.path.exists(model_path):
        st.info("Downloading model... (This may take a few minutes)")
        gdown.download(model_url, model_path, quiet=False)

# Load the trained model with caching
@st.cache_resource
def load_model():
    download_model()
    return tf.keras.models.load_model(model_path)

# Load model
model = load_model()

# Define class labels
class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Function to preprocess the image
def preprocess_image(img):
    img = img.convert("RGB")  # Ensure image is in RGB mode
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit UI
st.title("🧠 Brain Tumor Classification using ResNet50")
st.write("Upload an MRI scan image to detect the presence of a brain tumor.")

# File uploader
uploaded_file = st.file_uploader("Upload an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded MRI Scan", use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(img)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100

    # Display results
    st.subheader("📌 Prediction Results:")
    st.success(f"**Tumor Type:** {class_labels[predicted_class]}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # Display class probabilities with a bar chart
    st.subheader("📊 Class Probabilities:")
    prob_df = pd.DataFrame({"Class": class_labels, "Probability (%)": (prediction[0] * 100)})
    st.bar_chart(prob_df.set_index("Class"))

    st.write("👨‍⚕️ **Note:** This is an AI-based model and should not replace professional medical diagnosis.")
