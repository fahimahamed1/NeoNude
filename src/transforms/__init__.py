"""
OpenCV image transforms used in the pipeline.

Phase 0: dress -> correct  (color correction)
Phase 2: mask  -> maskref  (mask refinement)
Phase 4: maskdet -> maskfin (mask finalization with body annotations)
"""

from .annotation import BodyPart

# Re-export public functions
from .correct import correct_color          # noqa: F401
from .maskref import create_maskref         # noqa: F401
from .maskfin import create_maskfin         # noqa: F401

__all__ = [
    "BodyPart",
    "correct_color",
    "create_maskref",
    "create_maskfin",
]
