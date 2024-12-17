import numpy as np

def calculate_snr(image, noisy_image):
    """
    Calculate the Signal-to-Noise Ratio (SNR).
    image: Original clean image (ground truth)
    noisy_image: Image with noise

    
    """
    signal_power = np.mean(image.astype(np.float32) ** 2)
    noise = noisy_image.astype(np.float32) - image.astype(np.float32)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    snr = 10 * np.log10(signal_power / noise_power)
    return snr
