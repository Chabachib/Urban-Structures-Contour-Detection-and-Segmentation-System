import cv2

def clahe_method(image, clip_limit=3.0, grid_size=(4,4)):
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced_image = clahe.apply(image)
    return enhanced_image