"""
Image transforms package for NeoNude.

Contains:
- Color correction
- Mask creation and refinement
- Body part annotations
"""

from .color import correct_color
from .mask import (
    create_clothing_mask_fallback,
    refine_mask,
    blend_images,
    calculate_mask_coverage,
)
from .annotation import BodyPart

__all__ = [
    "correct_color",
    "create_clothing_mask_fallback",
    "refine_mask",
    "blend_images",
    "calculate_mask_coverage",
    "BodyPart",
]
