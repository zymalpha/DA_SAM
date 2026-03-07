"""Convert projection-domain TIFF stacks into the PNG folder structure used by training scripts."""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff


def find_stack(stack_dir: str, prefix: str, sample: str):
    candidates = [
        f'{prefix}_{sample}_f32.tiff',
        f'{prefix}_{sample}_u8.tiff',
        f'{prefix}_{sample}.tiff',
    ]
    for name in candidates:
        path = Path(stack_dir) / name
        if path.exists():
            return path
    return None


def to_uint8_stack(stack: np.ndarray, normalize: str) -> np.ndarray:
    if stack.dtype == np.uint8:
        return stack
    stack = stack.astype(np.float32)
    if normalize == 'per-stack':
        vmin, vmax = float(stack.min()), float(stack.max())
        if vmax - vmin < 1e-12:
            return np.zeros_like(stack, dtype=np.uint8)
        return ((stack - vmin) / (vmax - vmin) * 255.0).clip(0, 255).astype(np.uint8)

    out = np.zeros_like(stack, dtype=np.uint8)
    for idx in range(stack.shape[0]):
        plane = stack[idx]
        vmin, vmax = float(plane.min()), float(plane.max())
        if vmax - vmin >= 1e-12:
            out[idx] = ((plane - vmin) / (vmax - vmin) * 255.0).clip(0, 255).astype(np.uint8)
    return out


def export_stack(stack_path: Path, out_dir: Path, sample: str, role: str, normalize: str):
    stack = tiff.imread(stack_path)
    stack_u8 = to_uint8_stack(stack, normalize)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(stack_u8.shape[0]):
        name = f'{sample}_{role}_{idx:04d}.png'
        cv2.imwrite(str(out_dir / name), stack_u8[idx])


def main():
    parser = argparse.ArgumentParser(description='Export TIFF stacks to PNG slice folders')
    parser.add_argument('--proj_img_dir', required=True, help='Directory containing raw projection TIFF stacks')
    parser.add_argument('--eq_dir', required=True, help='Directory containing equalized projection TIFF stacks')
    parser.add_argument('--proj_metal_dir', required=True, help='Directory containing coarse metal projection TIFF stacks')
    parser.add_argument('--label_dir', required=True, help='Directory containing metal trace label TIFF stacks')
    parser.add_argument('--out_dir', required=True, help='Output dataset root')
    parser.add_argument('--normalize', choices=['per-slice', 'per-stack'], default='per-slice', help='Float-to-uint8 normalization for raw/equalized stacks')
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    proj_root = out_root / 'proj_img'
    eq_root = out_root / 'Eq_proj_img'
    metal_root = out_root / 'proj_metal'
    label_root = out_root / 'metal_trace_label'

    raw_samples = sorted({p.name[len('proj_img_'):].replace('_f32.tiff', '').replace('.tiff', '') for p in Path(args.proj_img_dir).glob('proj_img_*.tiff')})
    if not raw_samples:
        raise FileNotFoundError(f'No proj_img_*.tiff stacks found under {args.proj_img_dir}')

    for sample in raw_samples:
        raw_stack = find_stack(args.proj_img_dir, 'proj_img', sample)
        eq_stack = find_stack(args.eq_dir, 'proj_img', sample)
        metal_stack = find_stack(args.proj_metal_dir, 'proj_metal', sample)
        label_stack = find_stack(args.label_dir, 'metal_trace_label', sample)

        if raw_stack is None or eq_stack is None or metal_stack is None or label_stack is None:
            missing = [
                key for key, value in {
                    'proj_img': raw_stack,
                    'Eq_proj_img': eq_stack,
                    'proj_metal': metal_stack,
                    'metal_trace_label': label_stack,
                }.items() if value is None
            ]
            raise FileNotFoundError(f'Missing stacks for {sample}: {missing}')

        export_stack(raw_stack, proj_root / sample, sample, 'proj', args.normalize)
        export_stack(eq_stack, eq_root / sample, sample, 'eq', args.normalize)
        export_stack(metal_stack, metal_root / sample, sample, 'metal', 'per-stack')
        export_stack(label_stack, label_root / sample, sample, 'label', 'per-stack')
        print(f'Exported PNG slices for {sample}')


if __name__ == '__main__':
    main()
