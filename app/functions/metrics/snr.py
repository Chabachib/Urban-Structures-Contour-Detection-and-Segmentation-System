import numpy as np

def snr(image, noise_sigma):
    """
    Calculates Signal-to-Noise Ratio (SNR).
    Args:
        image (numpy.ndarray): Input image (grayscale).
        noise_sigma (float): Standard deviation of the noise.
    """
    mean_signal = np.mean(image / 255.0)  # Normalize to [0, 1]
    snr = 10 * np.log10(mean_signal**2 / (noise_sigma**2 + 1e-10))  # Avoid division by zero
    print(f"SNR: {snr:.2f} dB")
