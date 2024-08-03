
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the pre-trained model
model = load_model(r"Leaf_Disease_Detection_/blob/main/model/9.keras")
data_cat = ['Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy']

# Streamlit UI
st.markdown("<h1 style='text-align: center; font-family: Italic; color: green;'>Leaf Disease Detection</h1>", unsafe_allow_html=True)
st.sidebar.title("Project Overview")
st.sidebar.info(
    """
    *Overview:*
    The Leaf Disease Detection project leverages deep learning to automatically identify and classify diseases in plant leaves from images. This innovative approach aims to support agricultural practices by providing timely and accurate disease detection.

    *Objectives:*
    - *Automate Disease Detection:* Build a system that accurately classifies leaf diseases using deep learning.
    - *Enhance Crop Management:* Offer a tool for farmers to monitor plant health effectively.
    - *Reduce Chemical Usage:* Minimize pesticide use through early disease detection.

    *Impact:*
    This project showcases the potential of AI to revolutionize agriculture, improve crop yields, and reduce environmental impacts by enabling more precise and sustainable farming practices.
    """
)

# Add background image (ensure the images are publicly accessible or stored correctly)
st.markdown(
    """
    <style>
    header[data-testid="stHeader"] {
        background-image: url(https://www.shutterstock.com/image-vector/herbal-minimalist-vector-banner-hand-260nw-2143979437.jpg);
        background-repeat: repeat;
        background-size: contain;
        height: 10%;
    }
    section[data-testid="stSidebar"] {
        top: 10%;
    }
    footer[data-testid="stFooter"] {
        background-image: url(https://www.shutterstock.com/image-vector/herbal-minimalist-vector-banner-hand-260nw-2143979437.jpg);
        background-repeat: repeat;
        background-size: contain;
        height: 10%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Define a data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomFlip("horizontal"),
    layers.RandomContrast(0.2),
])

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    img_width = 226
    img_height = 226

    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = image.resize((img_width, img_height))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_batch = tf.expand_dims(image_array, 0)

    # Apply data augmentation
    augmented_image_batch = data_augmentation(image_batch)

    # Make predictions on the augmented image
    predictions = model.predict(augmented_image_batch)

    # Get the predicted class and confidence
    predicted_class = data_cat[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Display the results
    st.write('Predicted Disease: ' + predicted_class)
    st.write('Confidence: {:.2f}%'.format(confidence * 100))

    # Plot the augmented image with prediction
    augmented_image_np = augmented_image_batch[0].numpy().astype("uint8")
    fig, ax = plt.subplots()
    ax.imshow(augmented_image_np)
    ax.axis("off")
    ax.set_title(f'Predicted: {predicted_class}\nConfidence: {confidence * 100 :.2f} %', fontsize=12, color='red', fontweight='bold')
    st.pyplot(fig)
