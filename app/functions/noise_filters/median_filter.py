import cv2

def median_filter(image, kernel_size=5):
    """
    Apply a median filter to the image.
    Args:
        image: Input image (numpy array).
        kernel_size: Size of the kernel (odd integer).
    Returns:
        Smoothed image.
    """
    return cv2.medianBlur(image, kernel_size)
