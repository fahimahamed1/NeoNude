"""
Model caching utilities.

Handles downloading and caching models to local checkpoints directory.
"""

import os
from pathlib import Path


# Base checkpoints directory
CHECKPOINTS_DIR = Path(__file__).parent.parent.parent / "checkpoints"


def get_model_path(model_id: str) -> Path:
    """Get local path for a model, creating directory if needed.
    
    Converts HuggingFace model IDs like "user/model-name" to
    local paths like "checkpoints/user_model-name".
    
    Args:
        model_id: HuggingFace model identifier (e.g., "Lykon/dreamshaper-8-inpainting")
        
    Returns:
        Path to local model directory
    """
    # Replace "/" with "_" for safe directory name
    safe_name = model_id.replace("/", "_")
    model_path = CHECKPOINTS_DIR / safe_name
    
    # Create directory if it doesn't exist
    model_path.mkdir(parents=True, exist_ok=True)
    
    return model_path


def is_model_cached(model_id: str) -> bool:
    """Check if model is already cached locally.
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        True if model directory exists and has files
    """
    model_path = get_model_path(model_id)
    
    if not model_path.exists():
        return False
    
    # Check if directory has any files (not just empty directory)
    files = list(model_path.iterdir())
    return len(files) > 0
