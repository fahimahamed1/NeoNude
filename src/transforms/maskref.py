"""
Phase 2: Mask refinement.

Creates maskref by isolating the green clothing mask from the GAN output
and compositing it onto the corrected image.
"""

import cv2
import numpy as np


def create_maskref(cv_mask, cv_correct):
    """Create the refined mask (maskref) from the raw GAN mask.

    Extracts the green clothing region from the GAN mask and overlays it
    onto the color-corrected input image.

    Args:
        cv_mask: Raw mask image from the correct-to-mask GAN phase.
        cv_correct: Color-corrected input image.

    Returns:
        maskref image (512x512 BGR).
    """
    # Solid green background
    green = np.zeros((512, 512, 3), np.uint8)
    green[:, :, :] = (0, 255, 0)  # BGR

    # Filter for green pixels from the GAN mask
    f1 = np.asarray([0, 250, 0])
    f2 = np.asarray([10, 255, 10])
    green_mask = cv2.inRange(cv_mask, f1, f2)

    # Dilate the mask slightly
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.dilate(green_mask, kernel, iterations=1)

    # Composite: keep corrected image outside green, green inside
    green_mask_inv = cv2.bitwise_not(green_mask)
    res1 = cv2.bitwise_and(cv_correct, cv_correct, mask=green_mask_inv)
    res2 = cv2.bitwise_and(green, green, mask=green_mask)

    return cv2.add(res1, res2)
