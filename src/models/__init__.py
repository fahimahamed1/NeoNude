"""
Models package for NeoNude.

Contains:
- InpaintingModel: Stable Diffusion inpainting
- ClothingSegmenter: SegFormer clothing detection
- models: Model identifiers and paths
"""

from .models import (
    INPAINTING_MODEL,
    SEGMENTATION_MODEL,
    CLOTHING_LABELS,
)
from .inpainting import InpaintingModel
from .segmentation import ClothingSegmenter

__all__ = [
    "InpaintingModel",
    "ClothingSegmenter",
    "INPAINTING_MODEL",
    "SEGMENTATION_MODEL",
    "CLOTHING_LABELS",
]
