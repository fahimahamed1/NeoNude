"""
Body part annotation data class used by the mask finalization step.
"""


class BodyPart:
    """Represents a detected body part with bounding box, center, and dimensions."""

    def __init__(self, name, xmin, ymin, xmax, ymax, x, y, w, h):
        self.name = name
        # Bounding box
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        # Center
        self.x = x
        self.y = y
        # Dimensions
        self.w = w
        self.h = h
