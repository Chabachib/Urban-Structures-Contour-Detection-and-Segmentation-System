import numpy as np

def detect_salt_and_pepper(image, threshold=0.01):
    """
    Detects salt-and-pepper noise by counting extreme outlier pixels.
    Args:
        image (numpy.ndarray): Input image (grayscale).
        threshold (float): Pixel intensity threshold to detect outliers.
    """
    image_normalized = image.astype(np.float32) / 255.0
    extreme_pixels = np.logical_or(image_normalized <= threshold, 
                                   image_normalized >= 1 - threshold)
    salt_pepper_ratio = np.sum(extreme_pixels) / image_normalized.size
    if salt_pepper_ratio > 0.01:
        print(f"Salt-and-pepper noise detected: {salt_pepper_ratio * 100:.2f}% of pixels affected.")
    else:
        print("No significant salt-and-pepper noise detected.")