import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# Model Prediction Function
# -------------------------------
def model_prediction(test_image):
    model = tf.keras.models.load_model('final_model.keras')  # your trained model

    expected_size = model.input_shape[1:3]  # (224, 224)

    image = Image.open(test_image).convert("RGB")
    image = image.resize(expected_size)

    img_array = np.array(image)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    return prediction  # probability


# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Deepfake Detection Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Deepfake Detection"])

# -------------------------------
# HOME PAGE
# -------------------------------
if app_mode == "Home":
    st.header("Deepfake Image Detection System")
    st.markdown("""
    Welcome to the **Deepfake Detection System**!  
    This tool helps identify whether an uploaded image is **Real** or **Deepfake** using a deep learning model.

    ### 🔍 How It Works
    1. Upload any face image  
    2. System analyzes it using a trained EfficientNetB0 model  
    3. You receive prediction: **Real** or **Fake**

    ### 🚀 Why Use This App?
    - Detect manipulated images
    - Lightweight & fast deep learning model
    - Useful for media verification & research

    Go to the **Deepfake Detection** page to begin!
    """)

# -------------------------------
# ABOUT PAGE
# -------------------------------
elif app_mode == "About":
    st.header("About This Deepfake Detection Project")
    st.markdown("""
    ### 📌 Dataset Information
    This project uses the **Deepfake Image Detection Dataset** from Kaggle.

    It contains:
    - **Real face images**
    - **AI-generated / manipulated deepfake images**

    ### 🧠 Model Used
    The system uses **EfficientNetB0 Transfer Learning**, trained with:
    - 224×224 RGB inputs  
    - Binary classification (Real vs Fake)  
    - Sigmoid output layer  
    - Adam optimizer  
    - Data augmentation

    ### 🎯 Objective
    To build a fast and reliable system that automatically detects deepfake images with high accuracy.
    """)

# -------------------------------
# PREDICTION PAGE
# -------------------------------
elif app_mode == "Deepfake Detection":
    st.header("Deepfake Image Recognition")

    test_image = st.file_uploader("Upload Face Image:", type=["jpg", "jpeg", "png"])

    # Show Image
    if st.button("Show Image"):
        if test_image is not None:
            st.image(test_image, use_column_width=True)
        else:
            st.warning("Please upload an image first.")

    # Predict
    if st.button("Predict"):
        if test_image is not None:
            with st.spinner("Analyzing Image..."):
                pred = model_prediction(test_image)

                # Real vs Fake
                label = "Fake" if pred >= 0.5 else "Real"
                confidence = float(pred if pred >= 0.5 else 1 - pred)

                st.success(f" Prediction: **{label}**")
                st.info(f"Confidence Score: **{confidence:.4f}**")
        else:
            st.error("Upload an image to classify!")
