# Checkpoints

Place the trained model weight files in this directory.

## Required Files

| File | Phase | Description |
|------|-------|-------------|
| `cm.lib` | correct → mask | Clothing mask generation model (~700 MB) |
| `mm.lib` | maskref → maskdet | Anatomical detail detection model (~700 MB) |
| `mn.lib` | maskfin → nude | Final image generation model (~700 MB) |

## Setup

```bash
# After downloading, place them here:
checkpoints/
├── cm.lib
├── mm.lib
└── mn.lib
```

> These files are not included in the repository due to their large size.
