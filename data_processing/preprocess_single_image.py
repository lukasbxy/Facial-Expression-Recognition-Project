#!/usr/bin/env python3
"""
Script to preprocess a single image before we feed it into the model.

How to use (from project root):
    python data_processing/preprocess_single_image.py image.jpg
    python data_processing/preprocess_single_image.py --output ./results image.jpg
    python data_processing/preprocess_single_image.py --numpy image.jpg

It resizes to 64x64, converts to RGB, and normalizes to 0-1.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

# settings
TARGET_SIZE = (64, 64)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


class ImagePreprocessor:
    # simple class for prepping images (no face detection here)

    def __init__(self):
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID_SIZE
        )


    def grayscale_to_rgb(self, image):
        # need 3 channels for the model even if it's gray
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def apply_clahe(self, image):
        # fix lighting issues
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def resize_and_normalize(self, image):
        # standard resize and scale to 0-1
        resized = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def process_image(self, image_path):
        # run through the whole pipeline
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None, "Failed to load image"

            # step by step pipeline
            rgb = self.grayscale_to_rgb(image)
            enhanced = self.apply_clahe(rgb)
            normalized = self.resize_and_normalize(enhanced)

            return normalized, "Success"

        except Exception as e:
            return None, f"Error: {str(e)}"


def save_processed_image(processed, output_path, as_numpy=False):
    # save result as PNG or npy file
    if as_numpy:
        np.save(output_path.with_suffix('.npy'), processed)
    else:
        image_uint8 = (processed * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path.with_suffix('.png')), image_bgr)


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess images with basic pipeline'
    )
    parser.add_argument('images', nargs='+', help='Input image paths')
    parser.add_argument('--output', '-o', default='./preprocessed_output',
                       help='Output directory')
    parser.add_argument('--numpy', action='store_true',
                       help='Save as .npy instead of PNG')
    parser.add_argument('--visualize', action='store_true',
                       help='Show before/after visualization')
    args = parser.parse_args()

    # check if files actually exist
    image_paths = [Path(p) for p in args.images]
    for path in image_paths:
        if not path.exists():
            print(f"Image not found: {path}")
            return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("Basic Image Preprocessing Pipeline")
    print("="*60)
    processor = ImagePreprocessor()

    # process the images one by one
    results = []
    for img_path in image_paths:
        print(f"\nProcessing: {img_path.name}... ", end='')
        processed, status = processor.process_image(img_path)

        if processed is not None:
            output_path = output_dir / f"{img_path.stem}_preprocessed"
            save_processed_image(processed, output_path, args.numpy)
            ext = '.npy' if args.numpy else '.png'
            print(f"OK -> {output_path.with_suffix(ext)}")
            results.append((img_path, processed, True))
        else:
            print(f"FAILED: {status}")
            results.append((img_path, None, False))

    # show how it went
    successful = sum(1 for *_, ok in results if ok)
    print(f"\n{'='*60}")
    print(f"Done: {successful}/{len(results)} successful")
    print(f"Output: {output_dir} (64x64 RGB, normalized [0,1])")
    print("="*60)

    # plot results if requested
    if args.visualize and successful > 0:
        try:
            import matplotlib.pyplot as plt
            viz_results = [(p, img) for p, img, ok in results if ok]
            n = len(viz_results)
            fig, axes = plt.subplots(n, 2, figsize=(8, 3*n))
            if n == 1:
                axes = axes.reshape(1, -1)

            for idx, (orig_path, processed) in enumerate(viz_results):
                original = cv2.cvtColor(cv2.imread(str(orig_path)), cv2.COLOR_BGR2RGB)
                axes[idx, 0].imshow(original)
                axes[idx, 0].set_title(f'Original: {orig_path.name}')
                axes[idx, 0].axis('off')
                axes[idx, 1].imshow(processed)
                axes[idx, 1].set_title('Preprocessed (64x64)')
                axes[idx, 1].axis('off')

            plt.tight_layout()
            viz_path = output_dir / 'visualization.png'
            plt.savefig(viz_path, dpi=150)
            print(f"Visualization: {viz_path}")
            plt.show()
        except ImportError:
            print("Matplotlib not installed. Skipping visualization.")


if __name__ == "__main__":
    main()
