"""
Prompt templates for image generation.

Contains positive and negative prompts used for inpainting.
"""


class Prompts:
    """Prompt templates for the pipeline.
    
    Manages positive and negative prompts for different generation tasks.
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
        
        self._prompts = {
            "clothing_removal": (
                "nude naked woman, realistic skin, natural body, professional photo"
            ),
            "negative": (
                "clothes, clothing, dressed, bra, underwear, shirt, pants, dress, "
                "low quality, deformed, blurry, artifacts"
            ),
        }
    
    def get(self, prompt_type: str) -> str:
        """Get prompt by type.
        
        Args:
            prompt_type: Key from prompts dictionary.
            
        Returns:
            Prompt string, or empty string if not found.
        """
        return self._prompts.get(prompt_type, "")
    
    def set(self, prompt_type: str, prompt: str):
        """Set a custom prompt.
        
        Args:
            prompt_type: Key for the prompt.
            prompt: Prompt string.
        """
        self._prompts[prompt_type] = prompt
    
    def all(self) -> dict:
        """Get all prompts as dictionary."""
        return self._prompts.copy()


# Global prompts instance
prompts = Prompts()
