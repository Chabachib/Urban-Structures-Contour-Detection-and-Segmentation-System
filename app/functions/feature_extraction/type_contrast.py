import cv2
import numpy as np

def type_contrast(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    std_dev = np.std(img)

    print(f"Standard Deviation (Contrast Metric): {std_dev:.2f}")
    if std_dev < 50:
        print("The image has low contrast.")
    else:
        print("The image has high contrast.")

    return std_dev