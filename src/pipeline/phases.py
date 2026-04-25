"""
Pipeline phases for image transformation.

Individual processing steps used by the main pipeline.
"""

import cv2
import numpy as np
from typing import Optional

from ..models import ClothingSegmenter
from ..transforms import (
    correct_color,
    create_clothing_mask_fallback,
    refine_mask,
    calculate_mask_coverage,
)


class PhaseProcessor:
    """Handles individual processing phases of the pipeline.
    
    Separates phase logic from pipeline orchestration for cleaner code.
    """
    
    def __init__(self):
        """Initialize the phase processor."""
        self._segmenter = None
    
    @property
    def segmenter(self) -> ClothingSegmenter:
        """Lazy load the segmentation model."""
        if self._segmenter is None:
            self._segmenter = ClothingSegmenter()
        return self._segmenter
    
    def color_correction(self, image: np.ndarray) -> np.ndarray:
        """Phase 1: Apply color correction to the image.
        
        Args:
            image: BGR image.
            
        Returns:
            Color-corrected BGR image.
        """
        return correct_color(image)
    
    def detect_clothing(self, image: np.ndarray) -> np.ndarray:
        """Phase 2: Detect clothing regions in the image.
        
        Uses SegFormer for primary detection, falls back to
        color-based detection if model fails.
        
        Args:
            image: Color-corrected BGR image.
            
        Returns:
            Binary mask where white = clothing regions.
        """
        try:
            mask = self.segmenter.segment(image)
            
            # Check if valid results
            if np.sum(mask) < image.size * 0.001:
                print("[Pipeline] Segmentation found minimal clothing, using fallback...")
                mask = create_clothing_mask_fallback(image)
                
        except Exception as e:
            print(f"[Pipeline] Segmentation error: {e}")
            print("[Pipeline] Using fallback detection...")
            mask = create_clothing_mask_fallback(image)
        
        return mask
    
    def refine_clothing_mask(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Refine the clothing mask for better inpainting.
        
        Args:
            mask: Initial binary mask.
            image: Original image.
            
        Returns:
            Refined binary mask.
        """
        return refine_mask(mask, image)
    
    def check_mask_coverage(self, mask: np.ndarray) -> float:
        """Check the coverage percentage of the mask.
        
        Args:
            mask: Binary mask.
            
        Returns:
            Coverage percentage (0-100).
        """
        return calculate_mask_coverage(mask)
    
    def adjust_mask_if_needed(
        self,
        mask: np.ndarray,
        coverage: float,
        image: np.ndarray
    ) -> tuple:
        """Adjust mask if coverage is too large or small.
        
        Args:
            mask: Current binary mask.
            coverage: Current coverage percentage.
            image: Original image.
            
        Returns:
            Tuple of (adjusted_mask, adjusted_coverage).
        """
        if coverage > 70:
            print("[Pipeline] Mask too large, using fallback detection...")
            mask = create_clothing_mask_fallback(image)
            mask = refine_mask(mask, image)
            coverage = calculate_mask_coverage(mask)
        elif coverage < 5:
            print("[Pipeline] Warning: Very little clothing detected")
        
        return mask, coverage
