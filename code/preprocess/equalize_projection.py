"""Apply CLAHE to raw projection TIFF stacks."""
import argparse
import os
import re

import cv2
import numpy as np
from tifffile import TiffFile, imwrite
from tqdm import tqdm


def to_uint8_for_clahe(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    vmin, vmax = float(np.min(img)), float(np.max(img))
    if vmax - vmin < 1e-12:
        return np.zeros_like(img, dtype=np.uint8)
    return (np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0) * 255.0).astype(np.uint8)


def process_one_tif(in_path: str, out_path: str, clip_limit: float, tile_size: int):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    pages = []
    with TiffFile(in_path) as tif:
        for page in tqdm(tif.pages, desc=os.path.basename(in_path), unit='slice'):
            img = page.asarray()
            eq = clahe.apply(to_uint8_for_clahe(img))
            pages.append(eq)
    stack = np.stack(pages, axis=0).astype(np.uint8)
    imwrite(out_path, stack, imagej=True, compression=None, photometric='minisblack')


def main():
    parser = argparse.ArgumentParser(description='Equalize raw projection TIFF stacks with CLAHE')
    parser.add_argument('--in_dir', required=True, help='Directory containing proj_img_*.tiff stacks')
    parser.add_argument('--out_dir', required=True, help='Output directory for equalized TIFF stacks')
    parser.add_argument('--pattern', default=r'^proj_img_.*\.(tif|tiff)$', help='Regex for selecting input TIFF stacks')
    parser.add_argument('--clip_limit', type=float, default=2.0, help='OpenCV CLAHE clip limit')
    parser.add_argument('--tile_size', type=int, default=8, help='OpenCV CLAHE tile grid size')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    regex = re.compile(args.pattern, re.IGNORECASE)
    files = sorted([name for name in os.listdir(args.in_dir) if regex.match(name)])
    if not files:
        raise FileNotFoundError(f'No matching TIFF stacks found under {args.in_dir}')

    for name in files:
        process_one_tif(
            os.path.join(args.in_dir, name),
            os.path.join(args.out_dir, name),
            args.clip_limit,
            args.tile_size,
        )


if __name__ == '__main__':
    main()
