import cv2
import numpy as np

"""
    Apply a Gaussian filter to an image and return the filtered image along with the Gaussian kernel.

    Parameters:
    - image (np.ndarray): Input image (grayscale or RGB).
    - kernel_size (int): Size of the Gaussian kernel (must be odd and > 1).
    - sigma (float): Standard deviation for the Gaussian kernel.

    Returns:
    - filtered_image (np.ndarray): The image after applying the Gaussian filter.
    - kernel (np.ndarray): The Gaussian kernel used for filtering.
    """

def gaussian_filter(image: np.ndarray, kernel_size: int, sigma: float) -> tuple:
    if kernel_size % 2 == 0 or kernel_size <= 1:
        raise ValueError("Kernel size must be an odd integer greater than 1")

    # Generate the Gaussian kernel manually (optional for transparency)
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)  # Normalize the kernel

    # Apply Gaussian filter using OpenCV
    filtered_image = cv2.filter2D(image, -1, kernel)

    return filtered_image, kernel