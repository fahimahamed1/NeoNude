"""
Clothing segmentation model using SegFormer.

Uses the segformer_b2_clothes model for accurate clothing detection.
"""

import os
from PIL import Image
import numpy as np
import cv2
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from ..config import settings
from ..utils.output import suppress_output
from ..utils.model_cache import get_model_path, is_model_cached
from .models import SEGMENTATION_MODEL, CLOTHING_LABELS


class ClothingSegmenter:
    """Segment clothing regions from images using SegFormer.
    
    Uses the configured segmentation model from models.py which can detect:
    - Upper clothes (shirts, tops, jackets)
    - Skirts
    - Pants
    - Dresses
    - Belts
    """
    
    CLOTHING_LABELS = CLOTHING_LABELS
    
    def __init__(self, model_name: str = None):
        """Initialize the segmentation model.
        
        Args:
            model_name: HuggingFace model identifier. Defaults to SEGMENTATION_MODEL.
        """
        model_name = model_name or SEGMENTATION_MODEL
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self):
        """Load the segmentation model from cache or download."""
        local_path = get_model_path(self.model_name)
        
        if is_model_cached(self.model_name):
            print(f"[Segmenter] Loading from cache: {local_path}")
            load_path = str(local_path)
            # Suppress output when loading from cache
            with suppress_output():
                self.processor = SegformerImageProcessor.from_pretrained(load_path)
                self.model = SegformerForSemanticSegmentation.from_pretrained(load_path)
        else:
            print(f"[Segmenter] Downloading model...")
            print(f"[Segmenter] Model: {self.model_name}")
            load_path = self.model_name
            
            # Download with progress bar shown
            self.processor = SegformerImageProcessor.from_pretrained(load_path)
            self.model = SegformerForSemanticSegmentation.from_pretrained(load_path)
            
            # Save to local cache
            print(f"[Segmenter] Saving to: {local_path}")
            self.processor.save_pretrained(str(local_path))
            self.model.save_pretrained(str(local_path))
        
        self.model.to(settings.device)
        self.model.eval()
        print("[Segmenter] Model loaded!")
    
    def segment(self, image: np.ndarray) -> np.ndarray:
        """Segment clothing regions from image.
        
        Args:
            image: BGR image (numpy array).
            
        Returns:
            Binary mask where 255 = clothing, 0 = non-clothing.
        """
        # Convert BGR to RGB for PIL
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Preprocess image
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(settings.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Upscale logits to original image size
            logits = torch.nn.functional.interpolate(
                logits,
                size=pil_image.size[::-1],
                mode="bilinear",
                align_corners=False
            )
            
            # Get segmentation map
            segmentation = logits.argmax(dim=1)[0].cpu().numpy()
        
        # Create binary clothing mask
        clothing_mask = np.zeros(segmentation.shape, dtype=np.uint8)
        
        for label_id in self.CLOTHING_LABELS:
            clothing_mask[segmentation == label_id] = 255
        
        return clothing_mask
    
    def get_segmentation_map(self, image: np.ndarray) -> np.ndarray:
        """Get full segmentation map with all labels.
        
        Args:
            image: BGR image (numpy array).
            
        Returns:
            Segmentation map with label indices.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(settings.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=pil_image.size[::-1],
                mode="bilinear",
                align_corners=False
            )
            segmentation = logits.argmax(dim=1)[0].cpu().numpy()
        
        return segmentation
