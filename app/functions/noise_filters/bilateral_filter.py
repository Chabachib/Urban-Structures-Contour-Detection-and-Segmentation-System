import cv2

def bilateral_filter(image, diameter=15, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
