"""
Pipeline package for NeoNude.

Contains:
- NeoNudePipeline: Main transformation pipeline
- PhaseProcessor: Individual processing phases
"""

from typing import Optional
import numpy as np

from .core import NeoNudePipeline
from .phases import PhaseProcessor


def process(image: np.ndarray) -> Optional[np.ndarray]:
    """Run the transformation pipeline on an image.
    
    This is the main entry point for the pipeline.
    
    Args:
        image: Input BGR image (OpenCV/numpy array).
        
    Returns:
        Transformed BGR image, or None if failed.
    """
    pipeline = NeoNudePipeline()
    return pipeline.process(image)


__all__ = [
    "NeoNudePipeline",
    "PhaseProcessor",
    "process",
]
