import cv2
import numpy as np
import matplotlib.pyplot as plt

def type_contrast(image):

    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    std_dev = np.std(image)
    
    fig = plt.figure(figsize=(6, 4))
    plt.plot(hist)
    plt.title('Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    
    if std_dev < 50:
        contrast_message = f"Low contrast image (Standard Deviation: {std_dev:.2f})"
    else:
        contrast_message = f"High contrast image (Standard Deviation: {std_dev:.2f})"
    
    return std_dev, fig, contrast_message