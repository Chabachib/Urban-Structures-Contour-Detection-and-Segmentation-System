import numpy as np
from skimage.restoration import estimate_sigma

def noise_variance(image):
    """
    Estimates the variance of Gaussian noise in an image.
    Args:
        image (numpy.ndarray): Input image (grayscale).
    """
    image_normalized = image.astype(np.float32) / 255.0
    sigma = estimate_sigma(image_normalized, channel_axis=None)
    print(f"Noise Variance: {sigma**2:.2f}")
