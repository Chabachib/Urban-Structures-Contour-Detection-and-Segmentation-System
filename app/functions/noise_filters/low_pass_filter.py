import cv2
import numpy as np

def low_pass_filter(image, kernel_size=5):
    """
    Apply a low-pass filter to the image using a simple averaging filter.
    Args:
        image: Input image (numpy array).
        kernel_size: Size of the averaging kernel (odd integer).
    Returns:
        Smoothed image.
    """
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    return cv2.filter2D(image, -1, kernel)
