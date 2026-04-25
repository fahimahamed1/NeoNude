# NeoNude

<p align="center">
  <img src="https://img.shields.io/badge/Version-1.0.0-purple?style=flat-square" alt="Version 1.0.0">
  <img src="https://img.shields.io/badge/Python-3.7+-3776AB?style=flat-square&logo=python" alt="Python 3.7+">
  <img src="https://img.shields.io/badge/PyTorch-1.7+-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch 1.7+">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="MIT License">
</p>

<p align="center">
  <strong>GAN-based Image Transformation</strong><br>
  <em>A pix2pixHD architecture using a divide-et-impera approach for image transformation.</em>
</p>

---

## ✨ Features

- **Auto Resize** — Automatically resizes any input to 512x512 and restores original dimensions on output
- **3-Phase GAN Pipeline** — Clothing mask → Anatomical detection → Final generation
- **OpenCV Transforms** — Color correction, mask refinement, and mask finalization
- **pix2pixHD Architecture** — Global generator with residual blocks and instance normalization
- **Simple CLI** — Easy input/output with minimal arguments

---

<details>
<summary>📋 Requirements & Installation</summary>

### Requirements

| Component | Requirement |
|-----------|-------------|
| **Python** | 3.7 or higher |
| **PyTorch** | 1.7 or higher |
| **OpenCV** | 4.5+ (headless) |
| **Pillow** | 8.0+ |
| **NumPy** | 1.19+ |

### Installation

```bash
# Clone the repository
git clone https://github.com/fahimahamed1/NeoNude.git
cd NeoNude

# Install dependencies
pip install -r requirements.txt
```

### Model Checkpoints

Place the model weight files in the `checkpoints/` directory:

Download the required files using wget:
```bash
wget -O checkpoints/cm.lib https://huggingface.co/fahimahamed1/NeoNude/resolve/main/checkpoints/cm.lib
wget -O checkpoints/mm.lib https://huggingface.co/fahimahamed1/NeoNude/resolve/main/checkpoints/mm.lib
wget -O checkpoints/mn.lib https://huggingface.co/fahimahamed1/NeoNude/resolve/main/checkpoints/mn.lib
```
```
checkpoints/
├── cm.lib    (~700 MB)
├── mm.lib    (~700 MB)
└── mn.lib    (~700 MB)
```

</details>

<details>
<summary>📖 How to Use</summary>

### CLI

```bash
# Default: reads input.png, writes output.png
python main.py

# Custom input/output paths
python main.py -i photo.jpg -o result.png

# Help
python main.py --help
```

</details>

<details>
<summary>⚙️ Pipeline Architecture</summary>

Instead of a single network, the problem is split into 3 sub-problems:

1. **Mask generation** — Identifies clothing regions
2. **Anatomical attribute detection** — Produces an abstract body map
3. **Final image generation** — Creates the output from the refined mask

### Pipeline Phases

| Phase | Type | Description |
|-------|------|-------------|
| 0 | OpenCV | Color correction and normalization |
| 1 | GAN | Clothing mask generation (`cm.lib`) |
| 2 | OpenCV | Mask refinement |
| 3 | GAN | Anatomical detail detection (`mm.lib`) |
| 4 | OpenCV | Mask finalization with body annotations |
| 5 | GAN | Final image generation (`mn.lib`) |

</details>

<details>
<summary>📂 Project Structure</summary>

```
NeoNude/
├── main.py                      # CLI entry point
├── requirements.txt             # Python dependencies
├── README.md
├── src/                         # Core package
│   ├── __init__.py              # Package metadata
│   ├── config.py                # Pipeline configuration (Options)
│   ├── pipeline.py              # Pipeline orchestrator (process function)
│   ├── model.py                 # GAN model, dataset, utilities
│   └── transforms/              # OpenCV image transforms
│       ├── __init__.py
│       ├── annotation.py        # Body part data class
│       ├── correct.py           # Phase 0: color correction
│       ├── maskref.py           # Phase 2: mask refinement
│       └── maskfin.py           # Phase 4: mask finalization
└── checkpoints/                 # Model weight files (not tracked)
```

</details>

---

<details>
<summary>🛡️ Security Notes</summary>

⚠️ **This tool is intended for:**
- Personal use and educational purposes
- Research in GAN-based image transformation
- Academic study of pix2pixHD architecture

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
