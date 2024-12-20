import cv2

def median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)
