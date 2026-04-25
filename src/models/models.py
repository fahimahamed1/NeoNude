"""
Model identifiers and paths.

Central location for all HuggingFace model IDs.
Change these values to use different models.
"""

# =============================================================================
# INPAINTING MODELS
# =============================================================================

# Default inpainting model (Stable Diffusion 1.5 based)
INPAINTING_MODEL = "Lykon/dreamshaper-8-inpainting"

# Alternative inpainting models (uncomment to use):
# INPAINTING_MODEL = "runwayml/stable-diffusion-inpainting"
# INPAINTING_MODEL = "stabilityai/stable-diffusion-2-inpainting"
# INPAINTING_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

# =============================================================================
# SEGMENTATION MODELS
# =============================================================================

# Default clothing segmentation model
SEGMENTATION_MODEL = "mattmdjaga/segformer_b2_clothes"

# Alternative segmentation models (uncomment to use):
# SEGMENTATION_MODEL = "mattmdjaga/segformer_b3_clothes"
# SEGMENTATION_MODEL = "sayakpaul/segformer-b5-clothes"

# =============================================================================
# LABEL CONFIGURATION
# =============================================================================

# SegFormer clothing label indices
# 0=background, 1=hat, 2=hair, 3=sunglasses, 4=upper_clothes,
# 5=skirt, 6=pants, 7=dress, 8=belt, 9=left_shoe, 10=right_shoe,
# 11=face, 12=left_leg, 13=right_leg, 14=left_arm, 15=right_arm,
# 16=bag, 17=scarf
CLOTHING_LABELS = [4, 5, 6, 7, 8]  # upper_clothes, skirt, pants, dress, belt
