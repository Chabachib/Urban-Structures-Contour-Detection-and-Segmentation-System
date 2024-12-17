import numpy as np

def psnr(image, noisy_image):
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR).
    Args:
        image (numpy.ndarray): Original image (grayscale).
        noisy_image (numpy.ndarray): Noisy image (grayscale).
    """
    mse = np.mean((image.astype(np.float32) - noisy_image.astype(np.float32)) ** 2)
    if mse == 0:
        print("PSNR: Infinite (No noise)")
        return
    max_pixel = 255.0
    psnr = 10 * np.log10(max_pixel**2 / mse)
    print(f"PSNR: {psnr:.2f} dB")
    