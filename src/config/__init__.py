"""
Configuration package for NeoNude.

Contains:
- Settings: Device and generation parameters
- Prompts: Prompt templates for generation
"""

from .settings import Settings, settings, get_device_info
from .prompts import Prompts, prompts


class Config:
    """Combined configuration class for backward compatibility.
    
    Merges Settings and Prompts into a single interface.
    """
    
    def __init__(self):
        self._settings = settings
        self._prompts = prompts
    
    # Delegate to settings
    @property
    def device(self):
        return self._settings.device
    
    @property
    def dtype(self):
        return self._settings.dtype
    
    @property
    def num_inference_steps(self):
        return self._settings.num_inference_steps
    
    @property
    def guidance_scale(self):
        return self._settings.guidance_scale
    
    @property
    def strength(self):
        return self._settings.strength
    
    @property
    def target_size(self):
        return self._settings.target_size
    
    # Delegate to prompts
    def get_prompt(self, prompt_type: str) -> str:
        return self._prompts.get(prompt_type)
    
    def set_prompt(self, prompt_type: str, prompt: str):
        self._prompts.set(prompt_type, prompt)


# Legacy Options class for backward compatibility
Options = Config


__all__ = [
    "Config",
    "Options",
    "Settings",
    "settings",
    "Prompts",
    "prompts",
    "get_device_info",
]
