import cv2

def bilateral_filter(image, diameter=15, sigma_color=75, sigma_space=75):
    """
    Apply a bilateral filter to the image.
    Args:
        image: Input image (numpy array).
        diameter: Diameter of each pixel neighborhood.
        sigma_color: Filter sigma in the color space (range difference).
        sigma_space: Filter sigma in the coordinate space (spatial distance).
    Returns:
        Edge-preserving smoothed image.
    """
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
