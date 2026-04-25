"""
Stable Diffusion Inpainting model.

Uses DreamShaper 8 for high-quality image inpainting.
"""

import os
from PIL import Image
import numpy as np
import cv2
import torch
from diffusers import StableDiffusionInpaintPipeline

from ..config import settings
from ..utils.output import suppress_output, suppress_stdout
from ..utils.model_cache import get_model_path, is_model_cached
from .models import INPAINTING_MODEL


class InpaintingModel:
    """Stable Diffusion Inpainting model for image transformation.
    
    Uses the configured inpainting model from models.py for
    high-quality inpainting results.
    """
    
    DEFAULT_MODEL = INPAINTING_MODEL
    
    def __init__(self, model_name: str = None):
        """Initialize the inpainting model.
        
        Args:
            model_name: HuggingFace model identifier. Defaults to DreamShaper 8.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the inpainting pipeline from cache or download."""
        local_path = get_model_path(self.model_name)
        
        if is_model_cached(self.model_name):
            print(f"[Inpainter] Loading from cache: {local_path}")
            load_path = str(local_path)
            # Suppress output when loading from cache
            with suppress_output():
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    load_path,
                    torch_dtype=settings.dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
        else:
            print(f"[Inpainter] Downloading model...")
            print(f"[Inpainter] Model: {self.model_name}")
            load_path = self.model_name
            
            # Download with progress bar shown
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                load_path,
                torch_dtype=settings.dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            
            # Save to local cache
            print(f"[Inpainter] Saving to: {local_path}")
            self.pipeline.save_pretrained(str(local_path))
        
        # Handle memory management
        if settings.enable_cpu_offload:
            print("[Inpainter] Enabling CPU offload (slower, uses less VRAM)")
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(settings.device)
        
        # Enable memory efficient attention if available
        if hasattr(self.pipeline, 'enable_xformers_memory_efficient_attention'):
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        
        print("[Inpainter] Model loaded!")
    
    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str = None,
        negative_prompt: str = None,
        strength: float = None,
        num_inference_steps: int = None,
        guidance_scale: float = None,
    ) -> np.ndarray:
        """Perform inpainting on the masked region.
        
        Args:
            image: BGR image (numpy array).
            mask: Binary mask where white = inpaint region.
            prompt: Positive prompt for generation.
            negative_prompt: Negative prompt for generation.
            strength: Inpainting strength (0.0-1.0).
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG scale.
            
        Returns:
            Inpainted BGR image (numpy array).
        """
        # Use defaults from config
        strength = strength or settings.strength
        num_inference_steps = num_inference_steps or settings.num_inference_steps
        guidance_scale = guidance_scale or settings.guidance_scale
        
        # Default prompts
        if prompt is None:
            prompt = "nude naked woman, realistic skin, natural body, high quality"
        
        if negative_prompt is None:
            negative_prompt = "clothes, clothing, dressed, bra, underwear, low quality, deformed"
        
        # Convert BGR to RGB PIL
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image).convert("RGB")
        
        # Ensure mask is proper format
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        pil_mask = Image.fromarray(mask).convert("L")
        
        # Ensure image and mask are same size
        if pil_image.size != pil_mask.size:
            pil_mask = pil_mask.resize(pil_image.size, Image.NEAREST)
        
        # Run inpainting with suppressed progress bar
        with suppress_stdout():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=pil_image,
                mask_image=pil_mask,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=settings.get_generator(),
            ).images[0]
        
        # Convert back to BGR
        result_np = np.array(result)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
        
        return result_bgr
    
    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Allow calling the model directly."""
        return self.inpaint(*args, **kwargs)
