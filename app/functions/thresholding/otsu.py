import cv2

def otsu_threshold(image):

    otsu_threshold, threshold_image = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return threshold_image, otsu_threshold