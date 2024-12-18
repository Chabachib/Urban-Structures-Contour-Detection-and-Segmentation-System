import numpy as np
from skimage.restoration import estimate_sigma

def noise_variance(image):
    """
    Estimates the variance of Gaussian noise in an image.
    Args:
        image (numpy.ndarray): Input image (grayscale).
    Returns:
        str: Estimated noise variance
    """
    # Use original image values without normalization
    sigma = estimate_sigma(image, channel_axis=None)
    
    # Convert to percentage for better interpretation
    noise_level = (sigma / 255.0) * 100
    
    return f"Estimated Noise Level: {noise_level:.2f}%"
