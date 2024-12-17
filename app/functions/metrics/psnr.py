import numpy as np

def calculate_psnr(image, noisy_image):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    image: Original clean image (ground truth)
    noisy_image: Image with noise
    """
    mse = np.mean((image.astype(np.float32) - noisy_image.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr
