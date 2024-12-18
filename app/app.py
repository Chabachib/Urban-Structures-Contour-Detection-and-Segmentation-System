import streamlit as st
from PIL import Image
import numpy as np
from functions.noise_filters import bilateral_filter, low_pass_filter, median_filter, gaussian_filter
from functions.thresholding import adaptive_gaussian_filter

st.title("Welcome to the Urban Structure Segmentation System")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Add the "Segment" button
    if st.button("Segment"):
        # Convert the image to a NumPy array for processing
        image_array = np.array(image)

        # Apply filters (example sequence)
        image_filtered = median_filter(image_array,3)

        # Display the processed image
        st.image(image_filtered, caption="Segmented Image", use_column_width=True)
