"""
Color correction transforms.

Adjusts image contrast and color balance for better processing.
"""

import cv2
import math
import numpy as np


def correct_color(img: np.ndarray, percent: float = 5) -> np.ndarray:
    """Apply percentile-based color correction to an image.
    
    Clips extreme pixel values and normalizes each color channel
    independently to improve contrast and color balance.
    
    Args:
        img: BGR image (numpy array) with 3 channels.
        percent: Percentile range to clip (0-100).
        
    Returns:
        Color-corrected BGR image.
    
    Example:
        >>> corrected = correct_color(image, percent=5)
    """
    assert img.shape[2] == 3, "Image must have 3 channels"
    assert 0 < percent < 100, "Percent must be between 0 and 100"
    
    half_percent = percent / 200.0
    channels = cv2.split(img)
    out_channels = []
    
    for channel in channels:
        assert len(channel.shape) == 2, "Channel must be 2D"
        
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)
        
        # Sort to find percentiles
        flat = np.sort(flat)
        n_cols = flat.shape[0]
        
        low_val = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]
        
        # Apply threshold and normalize
        thresholded = _apply_threshold(channel, low_val, high_val)
        normalized = cv2.normalize(
            thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX
        )
        out_channels.append(normalized)
    
    return cv2.merge(out_channels)


def _apply_threshold(matrix: np.ndarray, low_value: float, high_value: float) -> np.ndarray:
    """Clip matrix values below low_value and above high_value."""
    low_mask = matrix < low_value
    matrix = _apply_mask(matrix, low_mask, low_value)
    high_mask = matrix > high_value
    matrix = _apply_mask(matrix, high_mask, high_value)
    return matrix


def _apply_mask(matrix: np.ndarray, mask: np.ndarray, fill_value: float) -> np.ndarray:
    """Fill masked positions with fill_value."""
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()
