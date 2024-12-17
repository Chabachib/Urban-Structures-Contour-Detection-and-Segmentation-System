import cv2

def estimate_noise_variance(noisy_image):
    """
    Estimate the noise variance of an image.
    noisy_image: Image with noise
    """
    # Convert to grayscale for simplicity
    gray = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
    noise_std = np.std(gray)
    return noise_std
