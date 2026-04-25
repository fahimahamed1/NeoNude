"""
NeoNude - Modern diffusion-based image transformation.

Uses Stable Diffusion inpainting with SegFormer clothing detection
for high-quality image-to-image transformation.
"""

from .utils import configure_quiet_mode

# Configure quiet mode on import
configure_quiet_mode()

from .pipeline import NeoNudePipeline, process
from .config import Config, Options, get_device_info
from .models import InpaintingModel, ClothingSegmenter

__version__ = "2.0.0"
__author__ = "Fahim Ahamed"
__description__ = "Modern diffusion-based image transformation"

__all__ = [
    "NeoNudePipeline",
    "process",
    "Config",
    "Options",
    "get_device_info",
    "InpaintingModel",
    "ClothingSegmenter",
]
