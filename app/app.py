import streamlit as st
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from functions import (
    gaussian_noise_variance,
    impulsive_noise_variance,
    type_contrast,
    compute_haralick_features,
    interpret_haralick,
    clahe_method,
    histogram_equalization,
)

st.set_page_config(layout="wide")
st.title("Welcome to the Urban Structure Segmentation System")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=250)

    if st.button("Segment"):
        image_array = np.array(image)
        grayscale_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        st.image(grayscale_image, caption="Grayscale Image", width=250)
        
        # Feature Extraction Section
        st.markdown("### Feature Extraction")
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Noise Analysis
            st.subheader("Noise Analysis")
            noise_result = gaussian_noise_variance(grayscale_image)
            sp_result = impulsive_noise_variance(grayscale_image)
            st.write(noise_result)
            st.write(sp_result)
            st.subheader("Texture Analysis")

        with col2:
            # Contrast Analysis
            st.subheader("Contrast Analysis")
            std_dev, fig, contrast_message = type_contrast(grayscale_image)
            st.write(contrast_message)
            st.pyplot(fig)
            plt.close(fig)