#!/usr/bin/env python3
"""
NeoNude - CLI entry point.

Modern diffusion-based image transformation using Stable Diffusion
inpainting with SegFormer clothing detection.

Usage:
    python main.py                          # Uses default input.png
    python main.py -i photo.jpg -o result.png
    python main.py -i input/ -o output/     # Batch processing
    python main.py -i photo.jpg --seed 42   # Reproducible results
"""

import sys
import argparse
import os
from pathlib import Path
import cv2

from src import process, NeoNudePipeline
from src.config import settings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NeoNude - Modern diffusion-based image transformation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                          # Process input.png -> output.png
    python main.py -i photo.jpg -o out.png  # Custom input/output
    python main.py -i input/ -o output/     # Batch process folder
    python main.py -i photo.jpg --seed 42   # Reproducible results
    python main.py -i photo.jpg --cpu-offload  # For low VRAM GPUs
    python main.py -i photo.jpg --steps 30 --quality fast  # Faster processing
        """
    )
    
    # Input/Output
    parser.add_argument(
        "-i", "--input",
        default="input.png",
        help="Input image path or folder (default: input.png)",
    )
    parser.add_argument(
        "-o", "--output",
        default="output.png",
        help="Output image path or folder (default: output.png)",
    )
    
    # Generation options
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results (default: random)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of inference steps (default: 50, higher = better quality)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="Guidance scale (default: 9.0, higher = more prompt adherence)",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help="Inpainting strength (default: 1.0, range: 0.0-1.0)",
    )
    
    # Quality presets
    parser.add_argument(
        "--quality",
        choices=["fast", "balanced", "quality"],
        default=None,
        help="Quality preset: fast (25 steps), balanced (50 steps), quality (80 steps)",
    )
    
    # Memory options
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable CPU offload for low VRAM GPUs (slower but uses less memory)",
    )
    
    # Debug options
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Save detected mask for debugging",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser.parse_args()


def apply_settings(args):
    """Apply CLI arguments to settings."""
    # Seed
    if args.seed is not None:
        settings.set_seed(args.seed)
        print(f"[Config] Seed: {args.seed}")
    
    # Memory
    if args.cpu_offload:
        settings.enable_cpu_offload = True
        print("[Config] CPU offload enabled")
    
    # Quality preset
    if args.quality:
        presets = {
            "fast": {"steps": 25, "guidance": 7.5},
            "balanced": {"steps": 50, "guidance": 9.0},
            "quality": {"steps": 80, "guidance": 10.0},
        }
        preset = presets[args.quality]
        settings.num_inference_steps = preset["steps"]
        settings.guidance_scale = preset["guidance"]
        print(f"[Config] Quality preset: {args.quality} ({preset['steps']} steps)")
    
    # Individual overrides
    if args.steps is not None:
        settings.num_inference_steps = args.steps
        print(f"[Config] Steps: {args.steps}")
    if args.guidance is not None:
        settings.guidance_scale = args.guidance
        print(f"[Config] Guidance: {args.guidance}")
    if args.strength is not None:
        settings.strength = args.strength
        print(f"[Config] Strength: {args.strength}")


def process_single(input_path: str, output_path: str, args) -> bool:
    """Process a single image.
    
    Args:
        input_path: Path to input image.
        output_path: Path to output image.
        args: CLI arguments.
        
    Returns:
        True if successful, False otherwise.
    """
    # Read input image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image '{input_path}'")
        return False
    
    print(f"Processing: {input_path}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    
    # Run pipeline
    pipeline = NeoNudePipeline()
    result = pipeline.process(img, save_mask=args.save_mask)
    
    if result is None:
        print("Error: Processing failed")
        return False
    
    # Save output
    cv2.imwrite(output_path, result)
    print(f"Done: {output_path}")
    return True


def process_batch(input_dir: str, output_dir: str, args) -> tuple:
    """Process all images in a directory.
    
    Args:
        input_dir: Path to input directory.
        output_dir: Path to output directory.
        args: CLI arguments.
        
    Returns:
        Tuple of (success_count, fail_count).
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image formats
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # Find all images
    input_path = Path(input_dir)
    images = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
    
    if not images:
        print(f"No images found in '{input_dir}'")
        return 0, 0
    
    print(f"Found {len(images)} images to process")
    
    success = 0
    fail = 0
    
    for i, img_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {img_path.name}")
        
        output_path = Path(output_dir) / f"output_{img_path.stem}.png"
        
        if process_single(str(img_path), str(output_path), args):
            success += 1
        else:
            fail += 1
    
    return success, fail


def main():
    """Main entry point."""
    args = parse_args()
    
    # Apply settings
    apply_settings(args)
    
    # Check if batch processing
    input_path = Path(args.input)
    
    if input_path.is_dir():
        # Batch processing
        output_dir = args.output
        if not Path(output_dir).suffix:  # It's a directory path
            success, fail = process_batch(args.input, output_dir, args)
            print(f"\nBatch complete: {success} success, {fail} failed")
            sys.exit(0 if fail == 0 else 1)
        else:
            print("Error: Output must be a directory when input is a directory")
            sys.exit(1)
    else:
        # Single image processing
        if process_single(args.input, args.output, args):
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
