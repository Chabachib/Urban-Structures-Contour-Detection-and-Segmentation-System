import cv2

def histogram_equalization(image):
     
    equalized_image = cv2.equalizeHist(image)
    return equalized_image