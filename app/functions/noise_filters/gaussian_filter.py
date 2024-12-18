import cv2
import numpy as np

import cv2
import numpy as np

def gaussian_filter(image, kernel_size, sigma):
    """
    Apply a Gaussian filter to the image.
    Args:
        image: Input image (numpy array).
        kernel_size: Size of the Gaussian kernel (odd integer).
        sigma: Standard deviation of the Gaussian kernel.
    Returns:
        Smoothed image.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
