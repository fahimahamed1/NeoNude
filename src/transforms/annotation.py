"""
Body part annotation data structures.

Used for representing detected body parts with bounding boxes.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class BodyPart:
    """Represents a detected body part with bounding box and dimensions.
    
    Attributes:
        name: Body part name (e.g., 'tit', 'aur', 'vag', 'belly').
        xmin: Left boundary of bounding box.
        ymin: Top boundary of bounding box.
        xmax: Right boundary of bounding box.
        ymax: Bottom boundary of bounding box.
        x: Center x coordinate.
        y: Center y coordinate.
        w: Width of the body part.
        h: Height of the body part.
    """
    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    x: float
    y: float
    w: float
    h: float
    
    @property
    def area(self) -> float:
        """Calculate area of the body part."""
        return self.w * self.h
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center coordinates as tuple."""
        return (self.x, self.y)
    
    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        """Get bounding box as (xmin, ymin, xmax, ymax)."""
        return (self.xmin, self.ymin, self.xmax, self.ymax)
