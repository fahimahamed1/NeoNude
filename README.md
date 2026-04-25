# NeoNude

<p align="center">
  <img src="https://img.shields.io/badge/Version-2.0.0-purple?style=flat-square" alt="Version 2.0.0">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="MIT License">
</p>

<p align="center">
  <strong>Diffusion-based Image Transformation</strong><br>
  <em>Using Stable Diffusion inpainting with SegFormer clothing detection.</em>
</p>

---

## ✨ Features

- **Stable Diffusion Inpainting** — High-quality image transformation using DreamShaper 8
- **SegFormer Detection** — Accurate clothing segmentation with label-based detection
- **Auto Model Download** — Models download from HuggingFace with progress bar
- **Local Model Cache** — Models saved to `checkpoints/` for offline use
- **Batch Processing** — Process entire folders of images at once
- **Seed Control** — Reproducible results with seed parameter
- **Quality Presets** — Fast (25 steps), Balanced (50 steps), Quality (80 steps)
- **CPU Offload** — Support for low VRAM GPUs
- **Mask Debug** — Save masks for troubleshooting
- **Modular Architecture** — Clean, well-organized codebase

---

<details>
<summary>📋 Requirements & Installation</summary>

### Requirements

| Component | Requirement |
|-----------|-------------|
| **Python** | 3.8 or higher |
| **PyTorch** | 2.0 or higher |
| **CUDA** | Recommended (GPU with 6GB+ VRAM) |
| **Diffusers** | 0.26+ |
| **Transformers** | 4.35+ |

### Installation

```bash
# Clone the repository
git clone https://github.com/fahimahamed1/NeoNude.git
cd NeoNude

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Models

Models are automatically downloaded from HuggingFace on first run:

```
checkpoints/
├── Lykon_dreamshaper-8-inpainting/    # ~2GB (Inpainting model)
├── mattmdjaga_segformer_b2_clothes/   # ~300MB (Segmentation model)
└── README.md
```

</details>

---

<details>
<summary>📖 How to Use</summary>

### Basic Usage

```bash
# Default: reads input.png, writes output.png
python main.py

# Custom input/output paths
python main.py -i photo.jpg -o result.png
```

### Batch Processing

```bash
# Process entire folder
python main.py -i input_folder/ -o output_folder/
```

### Quality Presets

```bash
# Fast processing (25 steps)
python main.py -i photo.jpg --quality fast

# Balanced (50 steps, default)
python main.py -i photo.jpg --quality balanced

# High quality (80 steps)
python main.py -i photo.jpg --quality quality
```

### Reproducible Results

```bash
# Use seed for consistent results
python main.py -i photo.jpg --seed 42
```

### Low VRAM GPUs

```bash
# Enable CPU offload (slower but uses less memory)
python main.py -i photo.jpg --cpu-offload
```

### Debug Mode

```bash
# Save mask for debugging
python main.py -i photo.jpg --save-mask
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input` | input.png | Input image path or folder |
| `-o, --output` | output.png | Output image path or folder |
| `--seed` | random | Random seed for reproducibility |
| `--steps` | 50 | Number of inference steps |
| `--guidance` | 9.0 | Guidance scale |
| `--strength` | 1.0 | Inpainting strength |
| `--quality` | balanced | Preset: fast, balanced, quality |
| `--cpu-offload` | off | Enable for low VRAM GPUs |
| `--save-mask` | off | Save mask for debugging |
| `-v, --verbose` | off | Enable verbose output |

</details>

---

<details>
<summary>⚙️ Pipeline Architecture</summary>

The pipeline runs in 3 phases:

### Pipeline Phases

| Phase | Component | Description |
|-------|-----------|-------------|
| 1 | OpenCV | Color correction and normalization |
| 2 | SegFormer | Detect clothing regions (labels: upper_clothes, skirt, pants, dress, belt) |
| 3 | Stable Diffusion | Generate result using inpainting |

### Models Used

| Model | Purpose | Size |
|-------|---------|------|
| `mattmdjaga/segformer_b2_clothes` | Clothing segmentation | ~300MB |
| `Lykon/dreamshaper-8-inpainting` | Image inpainting | ~2GB |

### Segmentation Labels

The SegFormer model detects 18 classes:

| Label | Name | Description |
|-------|------|-------------|
| 0 | background | Background pixels |
| 1 | hat | Headwear |
| 2 | hair | Hair region |
| 3 | sunglasses | Eye wear |
| 4 | upper_clothes | Shirts, tops, jackets |
| 5 | skirt | Skirts |
| 6 | pants | Pants, jeans |
| 7 | dress | Dresses |
| 8 | belt | Belts |
| 9-10 | shoes | Left/right shoe |
| 11 | face | Face region |
| 12-13 | legs | Left/right leg |
| 14-15 | arms | Left/right arm |
| 16 | bag | Bags |
| 17 | scarf | Scarves |

Default clothing labels: `[4, 5, 6, 7, 8]` (upper_clothes, skirt, pants, dress, belt)

</details>

---

<details>
<summary>📂 Project Structure</summary>

```
NeoNude/
├── main.py                     # CLI entry point
├── requirements.txt            # Python dependencies
├── README.md
├── src/
│   ├── __init__.py             # Package exports
│   ├── config/                 # Configuration
│   │   ├── __init__.py
│   │   ├── settings.py         # Device, seed, generation settings
│   │   └── prompts.py          # Prompt templates
│   ├── pipeline/               # Pipeline orchestrator
│   │   ├── __init__.py
│   │   ├── core.py             # Main pipeline class
│   │   └── phases.py           # Processing phases
│   ├── models/                 # Model implementations
│   │   ├── __init__.py
│   │   ├── models.py           # Model IDs (easy to change)
│   │   ├── inpainting.py       # Stable Diffusion inpainting
│   │   └── segmentation.py     # SegFormer clothing detection
│   ├── transforms/             # Image transformations
│   │   ├── __init__.py
│   │   ├── color.py            # Color correction
│   │   ├── mask.py             # Mask creation and refinement
│   │   └── annotation.py       # Body part data class
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── output.py           # Output suppression
│       └── model_cache.py      # Model caching
└── checkpoints/                # Cached models (auto-created)
```

</details>

---

<details>
<summary>🔧 Configuration</summary>

### Change Models

Edit `src/models/models.py`:

```python
# Inpainting models
INPAINTING_MODEL = "Lykon/dreamshaper-8-inpainting"
# INPAINTING_MODEL = "runwayml/stable-diffusion-inpainting"
# INPAINTING_MODEL = "stabilityai/stable-diffusion-2-inpainting"

# Segmentation models
SEGMENTATION_MODEL = "mattmdjaga/segformer_b2_clothes"
# SEGMENTATION_MODEL = "mattmdjaga/segformer_b3_clothes"

# Clothing labels to detect
CLOTHING_LABELS = [4, 5, 6, 7, 8]  # upper_clothes, skirt, pants, dress, belt
```

### Generation Settings

Edit `src/config/settings.py` or use CLI arguments:

```python
num_inference_steps = 50    # More steps = better quality
guidance_scale = 9.0        # Higher = more prompt adherence
strength = 1.0              # 1.0 = full transformation
seed = None                 # Set int for reproducibility
enable_cpu_offload = False  # True for low VRAM
```

### Custom Prompts

Edit `src/config/prompts.py`:

```python
prompts = {
    "clothing_removal": "nude naked woman, realistic skin, natural body",
    "negative": "clothes, clothing, dressed, bra, underwear, low quality",
}
```

</details>

---

<details>
<summary>🐍 API Usage</summary>

### Simple Usage

```python
from src import process
import cv2

image = cv2.imread("input.jpg")
result = process(image)
cv2.imwrite("output.jpg", result)
```

### With Custom Settings

```python
from src import NeoNudePipeline
from src.config import settings
import cv2

# Configure
settings.set_seed(42)  # Reproducible
settings.num_inference_steps = 30  # Faster

# Process
pipeline = NeoNudePipeline()
result = pipeline.process(image, save_mask=True)
```

### Use Individual Components

```python
from src.models import InpaintingModel, ClothingSegmenter
from src.transforms import correct_color, refine_mask
import cv2

# Load models
segmenter = ClothingSegmenter()
inpainter = InpaintingModel()

# Process step by step
image = cv2.imread("input.jpg")
corrected = correct_color(image)
mask = segmenter.segment(corrected)
mask = refine_mask(mask)
result = inpainter.inpaint(corrected, mask)
```

</details>

---

<details>
<summary>🛡️ Troubleshooting</summary>

### CUDA Out of Memory

```bash
# Option 1: Enable CPU offload
python main.py -i photo.jpg --cpu-offload

# Option 2: Use fewer steps
python main.py -i photo.jpg --steps 25

# Option 3: Use fast quality preset
python main.py -i photo.jpg --quality fast
```

### Model Download Issues

```bash
# Set HuggingFace token for faster downloads
huggingface-cli login
```

### Incomplete Clothing Removal

```bash
# Use quality preset with more steps
python main.py -i photo.jpg --quality quality

# Debug: check mask detection
python main.py -i photo.jpg --save-mask
```

### Slow Processing

- Ensure CUDA is available (`torch.cuda.is_available()`)
- Use `--quality fast` for quicker results
- CPU-only mode is significantly slower

</details>

---

<details>
<summary>⚠️ Security Notes</summary>

**This tool is intended for:**
- Personal use and educational purposes
- Research in diffusion-based image transformation
- Academic study of Stable Diffusion inpainting

**Do NOT use for:**
- Generating non-consensual intimate imagery
- Any form of harassment or exploitation
- Any illegal activities

> Always ensure proper authorization and ethical use before processing any images.

</details>

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Credits

- **DreamShaper** by [Lykon](https://huggingface.co/Lykon)
- **SegFormer** by [mattmdjaga](https://huggingface.co/mattmdjaga)
- **Stable Diffusion** by [Stability AI](https://stability.ai/)
- **Diffusers** by [HuggingFace](https://huggingface.co/)

---

## 👨‍💻 Author

**Fahim Ahamed**

[![GitHub](https://img.shields.io/badge/GitHub-fahimahamed1-181717?style=flat-square&logo=github)](https://github.com/fahimahamed1)

---

## ⭐ Support

If you find this project useful, please consider giving it a star! 🌟

---

<p align="center">
  Made with ❤️ for the open-source community
</p>
