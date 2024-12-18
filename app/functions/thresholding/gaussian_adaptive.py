import cv2
import numpy as np

def adaptive_gaussian_filter(image, max_kernel_size, max_sigma):
    """
    Apply an adaptive Gaussian filter to an image.

    Parameters:
        image (numpy.ndarray): Input grayscale or RGB image.
        max_kernel_size (int): Maximum size of the Gaussian kernel (should be odd).
        max_sigma (float): Maximum standard deviation for the Gaussian kernel.

    Returns:
        numpy.ndarray: Smoothed image after applying the adaptive Gaussian filter.
    """
    # Convert to grayscale if the image is in RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Compute the local variance using a sliding window
    kernel_size = max_kernel_size
    local_mean = cv2.blur(gray, (kernel_size, kernel_size))
    local_squared_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
    local_variance = local_squared_mean - local_mean**2

    # Normalize the variance to derive an adaptive sigma map
    normalized_variance = (local_variance - local_variance.min()) / (local_variance.max() - local_variance.min())
    sigma_map = 1 + normalized_variance * (max_sigma - 1)

    # Apply Gaussian filtering with the adaptive sigma map
    adaptive_filtered = np.zeros_like(gray, dtype=np.float32)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            sigma = sigma_map[i, j]
            kernel_size = int(2 * np.ceil(3 * sigma) + 1)  # Kernel size proportional to sigma
            kernel_size = min(max_kernel_size, max(3, kernel_size))  # Clamp kernel size
            gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)
            gaussian_filter = gaussian_kernel @ gaussian_kernel.T
            y_min, y_max = max(0, i - kernel_size // 2), min(gray.shape[0], i + kernel_size // 2 + 1)
            x_min, x_max = max(0, j - kernel_size // 2), min(gray.shape[1], j + kernel_size // 2 + 1)
            region = gray[y_min:y_max, x_min:x_max]
            weights = gaussian_filter[: region.shape[0], : region.shape[1]]
            adaptive_filtered[i, j] = np.sum(region * weights) / np.sum(weights)

    # Scale the output to the original range
    adaptive_filtered = np.clip(adaptive_filtered, 0, 255).astype(np.uint8)
    return adaptive_filtered
