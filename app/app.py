import streamlit as st
import cv2
from PIL import Image
import numpy as np
from functions.noise_filters import bilateral_filter, low_pass_filter, median_filter, gaussian_filter
from functions.thresholding import adaptive_gaussian_filter
from functions.feature_extraction import noise_variance, detect_salt_and_pepper, type_contrast

# page_bg_img = '''
# <style>
# .stApp {
#     background-image: url("https://your-image-url.jpg");
#     background-size: cover;
# }
# </style>
# '''

# st.markdown(page_bg_img, unsafe_allow_html=True)
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
        st.image(grayscale_image,caption="Grayscale Image", width=250)
        # ****************** Feature Extraction Section ******************
        st.markdown("### Feature Extraction")
        noise_result = noise_variance(grayscale_image)
        sp_result = detect_salt_and_pepper(grayscale_image)
        st.write(noise_result)
        st.write(sp_result)


