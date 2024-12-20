import numpy as np
from skimage.restoration import estimate_sigma

def impulsive_noise_variance(image):

    threshold = 0.01
    image_normalized = image.astype(np.float32) / 255.0
    extreme_pixels = np.logical_or(image_normalized <= threshold, image_normalized >= 1 - threshold)
    salt_pepper_ratio = np.sum(extreme_pixels) / image_normalized.size
    
    if salt_pepper_ratio > threshold:
        return f"Impulsive noise detected: {salt_pepper_ratio * 100:.2f}% of pixels affected."
    else:
        return "No significant impulsive noise detected."
    

def gaussian_noise_variance(image):
    
    sigma = estimate_sigma(image, channel_axis=None)
    noise_level = (sigma / 255.0) * 100
    return f"Estimated Gaussian Noise Level: {noise_level:.2f}%"
