from skimage.feature import graycomatrix, graycoprops
import numpy as np

def compute_haralick_features(image, distances=[1], angles=[0, 45, 90, 135]):
    """
    Compute Haralick texture features
    Args:
        image: Grayscale image
        distances: Array of pixel pair distances
        angles: Array of pixel pair angles in degrees
    Returns:
        dict: Haralick features
    """
    # Compute GLCM (Gray-Level Co-occurrence Matrix)
    glcm = graycomatrix(image, 
                        distances=distances,
                        angles=[x * np.pi / 180 for x in angles],
                        symmetric=True,
                        normed=True)
    
    # Calculate properties of GLCM
    features = {
        'contrast': graycoprops(glcm, 'contrast')[0, 0],     # Local intensity variation
        'dissimilarity': graycoprops(glcm, 'dissimilarity')[0, 0],  # Similar to contrast
        'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],  # Closeness of elements
        'energy': graycoprops(glcm, 'energy')[0, 0],        # Uniformity (angular second moment)
        'correlation': graycoprops(glcm, 'correlation')[0, 0],  # Linear dependency
        'ASM': graycoprops(glcm, 'ASM')[0, 0]              # Angular Second Moment
    }
    
    return features

def interpret_haralick(features):
    """
    Interpret Haralick features
    Args:
        features: Dictionary of Haralick features
    Returns:
        str: Interpretation of texture characteristics
    """
    interpretation = []
    
    # Contrast interpretation
    if features['contrast'] > 0.5:
        interpretation.append("High contrast texture with significant intensity variations")
    else:
        interpretation.append("Low contrast texture with subtle intensity variations")
    
    # Homogeneity interpretation
    if features['homogeneity'] > 0.75:
        interpretation.append("Very homogeneous texture")
    elif features['homogeneity'] > 0.5:
        interpretation.append("Moderately homogeneous texture")
    else:
        interpretation.append("Heterogeneous texture")
    
    # Energy/ASM interpretation
    if features['energy'] > 0.5:
        interpretation.append("Highly uniform texture")
    else:
        interpretation.append("Complex texture with many different patterns")
    
    return "\n".join(interpretation)