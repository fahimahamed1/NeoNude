"""
Main pipeline orchestrator for NeoNude.

Coordinates the complete image transformation process.
"""

import cv2
import numpy as np
from typing import Optional
from pathlib import Path

from ..config import Config, get_device_info
from ..models import InpaintingModel
from ..transforms import blend_images
from .phases import PhaseProcessor


class NeoNudePipeline:
    """Main pipeline for image transformation.
    
    Uses SegFormer for clothing detection and Stable Diffusion
    for inpainting to generate transformed images.
    
    Pipeline phases:
        1. Color correction
        2. Clothing detection and mask refinement
        3. Inpainting generation
    """
    
    def __init__(self):
        """Initialize the pipeline with lazy model loading."""
        self.config = Config()
        self.phases = PhaseProcessor()
        
        # Lazy-loaded models
        self._inpainter = None
        
        # Print device info
        device_info = get_device_info()
        print(f"[Pipeline] Device: {device_info['device']}")
        if device_info['cuda_available']:
            print(f"[Pipeline] GPU: {device_info['gpu_name']} ({device_info['vram_gb']} GB)")
    
    @property
    def inpainter(self) -> InpaintingModel:
        """Lazy load the inpainting model."""
        if self._inpainter is None:
            self._inpainter = InpaintingModel()
        return self._inpainter
    
    def process(self, image: np.ndarray, save_mask: bool = False) -> Optional[np.ndarray]:
        """Run the full transformation pipeline.
        
        Args:
            image: Input BGR image (OpenCV/numpy array), any size.
            save_mask: If True, save mask to 'mask_debug.png' for debugging.
            
        Returns:
            Transformed BGR image at original dimensions, or None if failed.
        """
        original_size = (image.shape[1], image.shape[0])
        print(f"[Pipeline] Input size: {original_size[0]}x{original_size[1]}")
        
        # Resize to target size
        processed = self._resize_input(image)
        
        # Phase 1: Color correction
        print("[Pipeline] Phase 1/3: Color correction")
        corrected = self.phases.color_correction(processed)
        
        # Phase 2: Detect and refine clothing mask
        print("[Pipeline] Phase 2/3: Detecting clothing...")
        mask = self._detect_and_refine_mask(corrected, save_mask=save_mask)
        
        # Phase 3: Generate result
        print("[Pipeline] Phase 3/3: Generating...")
        result = self._generate(corrected, mask)
        
        # Restore original dimensions
        if result is not None:
            result = self._resize_output(result, original_size)
        
        return result
    
    def _resize_input(self, image: np.ndarray) -> np.ndarray:
        """Resize input image to target size.
        
        Args:
            image: Input BGR image.
            
        Returns:
            Resized image.
        """
        original_size = (image.shape[1], image.shape[0])
        target = self.config.target_size
        
        if original_size != (target[1], target[0]):
            return cv2.resize(image, target, interpolation=cv2.INTER_AREA)
        return image.copy()
    
    def _resize_output(
        self,
        image: np.ndarray,
        original_size: tuple
    ) -> np.ndarray:
        """Resize output image back to original dimensions.
        
        Args:
            image: Processed BGR image.
            original_size: Original (width, height).
            
        Returns:
            Resized image.
        """
        target = self.config.target_size
        if original_size != (target[1], target[0]):
            return cv2.resize(image, original_size, interpolation=cv2.INTER_LINEAR)
        return image
    
    def _detect_and_refine_mask(self, image: np.ndarray, save_mask: bool = False) -> np.ndarray:
        """Detect clothing and create refined mask.
        
        Args:
            image: Color-corrected BGR image.
            save_mask: If True, save mask for debugging.
            
        Returns:
            Refined binary mask.
        """
        # Detect clothing
        mask = self.phases.detect_clothing(image)
        
        # Refine mask
        mask = self.phases.refine_clothing_mask(mask, image)
        
        # Check coverage
        coverage = self.phases.check_mask_coverage(mask)
        print(f"[Pipeline] Mask coverage: {coverage:.1f}%")
        
        # Adjust if needed
        mask, _ = self.phases.adjust_mask_if_needed(mask, coverage, image)
        
        # Save mask for debugging
        if save_mask:
            cv2.imwrite("mask_debug.png", mask)
            print("[Pipeline] Mask saved to: mask_debug.png")
        
        return mask
    
    def _generate(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """Generate the transformed image using inpainting.
        
        Args:
            image: Color-corrected BGR image.
            mask: Refined binary mask.
            
        Returns:
            Generated BGR image, or None if failed.
        """
        # Check for empty mask
        if np.sum(mask) == 0:
            print("[Pipeline] Warning: Empty mask, returning original")
            return image.copy()
        
        # Get prompts
        prompt = self.config.get_prompt("clothing_removal")
        negative_prompt = self.config.get_prompt("negative")
        
        # Run inpainting
        result = self.inpainter.inpaint(
            image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=self.config.strength,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
        )
        
        # Blend for smooth transition
        result = blend_images(image, result, mask)
        
        return result
