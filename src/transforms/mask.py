"""
Mask creation and refinement utilities.

Functions for creating and refining clothing masks for inpainting.
"""

import cv2
import numpy as np


def create_clothing_mask_fallback(image: np.ndarray) -> np.ndarray:
    """Create clothing mask using color-based skin detection.
    
    This is a fallback method when segmentation model fails.
    Uses HSV and YCrCb color spaces to detect skin, then inverts
    to find potential clothing regions.
    
    Args:
        image: BGR image.
        
    Returns:
        Binary mask where white = clothing regions.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Detect skin using multiple methods
    # Method 1: HSV skin detection
    skin_mask_hsv = cv2.inRange(hsv, np.array([0, 15, 50]), np.array([25, 170, 255]))
    
    # Method 2: YCrCb skin detection (better for various skin tones)
    skin_mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    
    # Combine skin masks
    skin_mask = cv2.bitwise_or(skin_mask_hsv, skin_mask_ycrcb)
    
    # Clean up skin mask
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    # Clothing is the inverse of skin
    clothing_mask = cv2.bitwise_not(skin_mask)
    
    # Additional color detection for obvious clothing colors
    # Dark colors (common in clothing)
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))
    
    # Saturated colors (clothing typically has more saturation than skin)
    sat_mask = cv2.inRange(hsv, np.array([0, 60, 40]), np.array([180, 255, 255]))
    
    # Combine masks
    clothing_mask = cv2.bitwise_or(clothing_mask, dark_mask)
    clothing_mask = cv2.bitwise_and(clothing_mask, sat_mask)
    
    # Final cleanup
    clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_OPEN, kernel)
    clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_CLOSE, kernel)
    
    return clothing_mask


def refine_mask(mask: np.ndarray, image: np.ndarray = None) -> np.ndarray:
    """Refine a clothing mask for better inpainting results.
    
    Dilates the mask to ensure clothing edges are covered,
    then applies Gaussian blur for smooth transitions.
    
    Args:
        mask: Initial binary mask.
        image: Original BGR image (unused, kept for API compatibility).
        
    Returns:
        Refined binary mask.
    """
    # Dilate mask to cover clothing edges
    kernel = np.ones((7, 7), np.uint8)
    refined = cv2.dilate(mask, kernel, iterations=2)
    
    # Smooth edges with Gaussian blur
    refined = cv2.GaussianBlur(refined, (11, 11), 0)
    
    # Threshold back to binary
    _, refined = cv2.threshold(refined, 100, 255, cv2.THRESH_BINARY)
    
    return refined


def blend_images(
    original: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """Blend generated result with original image using mask.
    
    Creates smooth transition between original and generated regions.
    
    Args:
        original: Original BGR image.
        generated: Generated BGR image.
        mask: Binary mask where white = generated region.
        
    Returns:
        Blended BGR image.
    """
    # Create smooth transition mask
    blur_mask = cv2.GaussianBlur(mask, (21, 21), 0)
    blur_mask = blur_mask.astype(np.float32) / 255.0
    
    # Expand dimensions for broadcasting
    blur_mask = np.expand_dims(blur_mask, axis=2)
    blur_mask = np.repeat(blur_mask, 3, axis=2)
    
    # Blend images
    blended = (original * (1 - blur_mask) + generated * blur_mask).astype(np.uint8)
    
    return blended


def calculate_mask_coverage(mask: np.ndarray) -> float:
    """Calculate the percentage of image covered by mask.
    
    Args:
        mask: Binary mask.
        
    Returns:
        Coverage percentage (0-100).
    """
    return np.sum(mask > 0) / mask.size * 100
