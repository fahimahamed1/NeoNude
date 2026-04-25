#!/usr/bin/env python3
"""
NeoNude - CLI entry point.

Usage:
    python main.py                          # Uses default input.png
    python main.py -i photo.jpg -o result.png
    python main.py --input photo.jpg --output result.png
"""

import sys
import argparse
import cv2

from src.pipeline import process


def parse_args():
    parser = argparse.ArgumentParser(
        description="NeoNude - GAN-based image transformation"
    )
    parser.add_argument(
        "-i", "--input",
        default="input.png",
        help="Path to input image (default: input.png, must be 512x512)",
    )
    parser.add_argument(
        "-o", "--output",
        default="output.png",
        help="Path to output image (default: output.png)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Read input
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Could not read image '{args.input}'")
        sys.exit(1)

    # Process
    print(f"Processing: {args.input}")
    result = process(img)

    if result is None:
        print("Error: Processing failed — no body parts detected.")
        sys.exit(1)

    # Write output
    cv2.imwrite(args.output, result)
    print(f"Done: {args.output}")
    sys.exit(0)


if __name__ == "__main__":
    main()
