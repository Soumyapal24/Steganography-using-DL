import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import cv2

# Load the trained model
MODEL_PATH = "/home/kudsit/Downloads/dl/Deep_steganography_model.h5"
model = load_model(MODEL_PATH)

# Define image size (64x64 assumed)
IMG_SHAPE = (64, 64)

# Function to preprocess image
def preprocess_image(image: Image.Image):
    """
    Preprocess an image for model input:
    - Resize to required dimensions
    - Normalize pixel values to [0, 1]
    - Add batch dimension
    """
    image = image.resize(IMG_SHAPE)
    image_array = img_to_array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to calculate pixel difference between two images
def calculate_pixel_difference(image1: np.array, image2: np.array):
    """
    Calculate the absolute pixel difference between two images.
    """
    diff = cv2.absdiff(image1, image2)  # Compute absolute difference
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    return diff

# Streamlit App
st.title("Steganography using Deep Learning")
st.write(
    "This application demonstrates a deep learning model for steganography. "
    "Upload a **cover image** and a **secret image** for encoding and decoding."
)

# Upload the cover image
cover_file = st.file_uploader("Upload the Cover Image", type=["jpg", "png", "jpeg"])

# Upload the secret image
secret_file = st.file_uploader("Upload the Secret Image", type=["jpg", "png", "jpeg"])

if cover_file and secret_file:
    try:
        # Convert uploaded files to PIL images
        cover_image = Image.open(cover_file).convert("RGB")
        secret_image = Image.open(secret_file).convert("RGB")

        # Display the uploaded images
        st.image(cover_image, caption="Cover Image", use_column_width=True)
        st.image(secret_image, caption="Secret Image", use_column_width=True)

        # Preprocess images
        cover_preprocessed = preprocess_image(cover_image)
        secret_preprocessed = preprocess_image(secret_image)

        # Perform prediction (encode)
        st.write("Encoding the images...")
        stego_image = model.predict([cover_preprocessed, secret_preprocessed])[0]  # Get the first batch result
        stego_image = (stego_image * 255).astype(np.uint8)  # Convert back to 0-255 range

        # Display the encoded image
        st.image(stego_image, caption="Stego Image (Encoded)", use_column_width=True)

        # Perform decoding (if applicable, requires a separate decoder model or functionality)
        # decoded_image = model.predict(stego_image)  # Example: requires decoding model
        # st.image(decoded_image, caption="Decoded Secret Image", use_column_width=True)

        # Calculate pixel loss (absolute difference between images)
        cover_np = np.array(cover_image.resize(IMG_SHAPE))  # Resize to match prediction size
        pixel_difference = calculate_pixel_difference(cover_np, stego_image)
        st.image(pixel_difference, caption="Pixel Difference (Loss)", use_column_width=True)

        # Optionally, display average pixel loss
        avg_pixel_loss = np.mean(pixel_difference)
        st.write(f"Average Pixel Loss: {avg_pixel_loss:.4f}")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

elif cover_file:
    st.warning("Please upload both a cover image and a secret image.")
else:
    st.info("Please upload a cover image to begin.")

# Sidebar Information
st.sidebar.title("About")
st.sidebar.info(
    """
    This application is a demonstration of a deep learning-based steganography model. 
    The project was developed by:

    - **Soumya Pal**: Roll No: 231043, Email: soumya.cs23@duk.ac.in
    - **Marcie M**: Roll No: 231030, Email: marcie.cs23@duk.ac.in
    """
)
