"""
Phase 0: Color correction.

Adjusts image contrast by clipping extreme pixel values and normalizing
each color channel independently.
"""

import cv2
import math
import numpy as np


def correct_color(img, percent=5):
    """Apply percentile-based color correction to an image.

    Args:
        img: BGR image (numpy array) with 3 channels.
        percent: Percentile range to clip (0-100).

    Returns:
        Color-corrected BGR image.
    """
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0
    channels = cv2.split(img)
    out_channels = []

    for channel in channels:
        assert len(channel.shape) == 2
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)
        assert len(flat.shape) == 1

        flat = np.sort(flat)
        n_cols = flat.shape[0]
        low_val = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]

        thresholded = _apply_threshold(channel, low_val, high_val)
        normalized = cv2.normalize(
            thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX
        )
        out_channels.append(normalized)

    return cv2.merge(out_channels)


def _apply_threshold(matrix, low_value, high_value):
    """Clip matrix values below low_value and above high_value."""
    low_mask = matrix < low_value
    matrix = _apply_mask(matrix, low_mask, low_value)
    high_mask = matrix > high_value
    matrix = _apply_mask(matrix, high_mask, high_value)
    return matrix


def _apply_mask(matrix, mask, fill_value):
    """Fill masked positions with fill_value."""
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()
