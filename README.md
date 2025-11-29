🚀 Deepfake Image Detection System

A complete end-to-end Deepfake Image Detection project using EfficientNetB0, TensorFlow/Keras, and a Streamlit web application.
This model classifies facial images as Real or Fake and provides confidence scores for real-time inference.

📌 Overview

Deepfake technology has rapidly advanced, making it easier to generate manipulated facial images. This project aims to build a reliable system that can automatically identify such deepfake images using deep learning and modern computer vision techniques.

The project includes:

Data preprocessing

Transfer learning model

Model training & fine-tuning

Real-time prediction with a Streamlit UI

Utilities for visualization and dataset reduction

📂 Dataset

Kaggle Dataset: Deepfake and Real Images
📎 https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

Classes included:

Fake → Deepfake/manipulated facial images

Real → Genuine facial images

🧠 Model Architecture

This project uses EfficientNetB0 with transfer learning.

✔ Base Model

EfficientNetB0 (ImageNet weights)

Base layers frozen for feature extraction

Later fine-tuned for improved accuracy

✔ Custom Classification Head

Global Average Pooling

Dense (128, ReLU)

Dropout (0.2–0.3)

Dense (1, Sigmoid)

✔ Training Specs

Optimizer: Adam

Loss: Binary Crossentropy

Metrics: Accuracy, AUC

Callbacks include:

EarlyStopping

ModelCheckpoint

ReduceLROnPlateau

📊 Evaluation

Performance metrics:

Accuracy: ~82–90%

AUC Score: 0.90+

Loss curves and Confusion Matrix included in notebooks

🖥️ Streamlit Web Application

This project includes a complete Streamlit UI for real-time deepfake detection.

Features:

Upload an image (jpg/png/jpeg)

View the uploaded image

Predict if it's Real or Fake

Display confidence score

Clean Home, About, and Detection pages
