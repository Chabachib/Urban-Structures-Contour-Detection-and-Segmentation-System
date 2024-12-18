import cv2
import numpy as np
import matplotlib.pyplot as plt

def type_contrast(image):
    """
    Analyzes image contrast and returns histogram plot data
    Args:
        image (numpy.ndarray): Input grayscale image
    Returns:
        tuple: (standard deviation, figure, contrast_message)
    """
    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    std_dev = np.std(image)
    
    # Create histogram plot
    fig = plt.figure(figsize=(6, 4))
    plt.plot(hist)
    plt.title('Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    
    # Prepare contrast message
    if std_dev < 50:
        contrast_message = f"Low contrast image (Standard Deviation: {std_dev:.2f})"
    else:
        contrast_message = f"High contrast image (Standard Deviation: {std_dev:.2f})"
    
    return std_dev, fig, contrast_message