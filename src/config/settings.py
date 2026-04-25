"""
Pipeline configuration settings.

Device detection, generation parameters, and processing settings.
"""

import torch
import random
import numpy as np


class Settings:
    """Global settings for the NeoNude pipeline.
    
    Handles device detection and generation parameters.
    Singleton pattern ensures consistent settings across the pipeline.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # Device settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Memory settings
        self.enable_cpu_offload = False  # For low VRAM GPUs
        
        # Generation settings
        self.num_inference_steps = 50
        self.guidance_scale = 9.0
        self.strength = 1.0
        
        # Seed for reproducibility
        self.seed = None  # None = random, or set int for reproducible results
        
        # Processing settings
        self.target_size = (512, 512)
    
    def set_seed(self, seed: int = None):
        """Set random seed for reproducibility.
        
        Args:
            seed: Integer seed, or None for random.
        """
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
    def get_generator(self):
        """Get torch generator for current seed.
        
        Returns:
            torch.Generator or None
        """
        if self.seed is not None:
            return torch.Generator(device=self.device).manual_seed(self.seed)
        return None


# Global settings instance
settings = Settings()


def get_device_info() -> dict:
    """Get information about the available compute device.
    
    Returns:
        Dictionary with device, cuda_available, and gpu_name.
    """
    info = {
        "device": "cpu",
        "cuda_available": False,
        "gpu_name": None,
        "vram_gb": None,
    }
    
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["cuda_available"] = True
        info["gpu_name"] = torch.cuda.get_device_name(0)
        # Get VRAM
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        info["vram_gb"] = round(vram_bytes / (1024**3), 1)
    
    return info
