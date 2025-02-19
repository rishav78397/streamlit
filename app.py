import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Path to your Keras model
model_path = 'model 2.keras'

# Load the Keras model
model = load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image, target_size):
    # Convert the image to grayscale (if your model expects grayscale images)
    image = image.convert('L')
    # Resize the image to the target size
    image = image.resize(target_size)
    # Convert image to numpy array and normalize
    image_array = np.array(image) / 255.0
    # Reshape the image to (1, target_size[0], target_size[1], 1) for grayscale
    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Streamlit app UI
st.title("Image Classification App")
st.write("Upload an image to classify it using the trained model.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the uploaded image
    image = Image.open(uploaded_file)
    target_size = (150, 150)  # Update this based on your model's input size
    image_array = preprocess_image(image, target_size)

    # Make prediction
    prediction = model.predict(image_array)

    # Convert the prediction into a readable output
    predicted_percentage = prediction[0] * 100
    predicted_class = "Class 1" if predicted_percentage > 50 else "Class 0"
    
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {predicted_percentage[0]:.2f}%")
