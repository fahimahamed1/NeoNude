# Checkpoints

This directory stores downloaded models locally.

## How It Works

Models are automatically downloaded from HuggingFace on first run and saved here for future use.

## Directory Structure

After first run, you'll see:

```
checkpoints/
├── Lykon_dreamshaper-8-inpainting/    # Inpainting model (~2GB)
│   ├── model_index.json
│   ├── scheduler/
│   ├── text_encoder/
│   ├── tokenizer/
│   ├── unet/
│   └── vae/
├── mattmdjaga_segformer_b2_clothes/   # Segmentation model (~300MB)
│   ├── config.json
│   └── model.safetensors
└── README.md
```

## Benefits

- **Faster loading**: Models load from local disk after first download
- **Offline use**: Works without internet after initial download
- **Portable**: Copy `checkpoints/` folder to another machine

## Changing Models

Edit `src/models/models.py` to use different models. New models will be downloaded and cached automatically.

## Manual Download

To pre-download models before running:

```bash
python -c "from src.models import InpaintingModel; InpaintingModel()"
python -c "from src.models import ClothingSegmenter; ClothingSegmenter()"
```

## Cleanup

To re-download models, simply delete the model folder and run again:

```bash
rm -rf checkpoints/Lykon_dreamshaper-8-inpainting/
rm -rf checkpoints/mattmdjaga_segformer_b2_clothes/
```
