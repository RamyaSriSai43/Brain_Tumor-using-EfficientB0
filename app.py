import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from gtts import gTTS
import os
import IPython.display as ipd
import pyttsx3

# Load the trained model
model = tf.keras.models.load_model("C:\\Users\\user\\OneDrive\\Desktop\\mp\\efficientnetB0.h5")


# Define the labels
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

# Define the function to classify the uploaded image
def classify_image(image):
    if image.shape[-1] == 4:
        # Convert RGBA image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    image = cv2.resize(image, (150, 150))  # Resize image to the required input size
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = tf.keras.applications.efficientnet.preprocess_input(image)  # Preprocess the image
    predictions = model.predict(image)  # Predict the class
    predicted_class = np.argmax(predictions)  # Get the class with the highest probability
    predicted_label = labels[predicted_class]  # Map the class index to the label
    confidence = predictions[0][predicted_class]  # Get the confidence of the prediction

    return predicted_label, confidence, predictions

# Streamlit interface
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Brain Tumor Detection using EfficientNetB0")
st.markdown('''
This app uses a pre-trained EfficientNetB0 model to classify MRI images of the brain into one of the following categories:
- **Glioma Tumor**
- **No Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**

Upload an MRI image to get started.
''')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.markdown("### Classifying...")

    # Classify the image
    label, confidence, predictions = classify_image(image)

    # Generate speech
    text=f"The predicted label is {label} with a confidence of {confidence*100:.2f} percent."
    # Display the results
    st.success(f"**Predicted Label:** {label}")
    st.info(f"**Confidence:** {confidence*100:.2f}%")

    # Display a matplotlib plot for confidence scores
    fig, ax = plt.subplots()
    ax.barh(labels, predictions[0], color='skyblue')
    ax.set_xlim([0, 1])
    ax.set_xlabel('Confidence')
    ax.set_title('Prediction Confidence')
    st.pyplot(fig)

