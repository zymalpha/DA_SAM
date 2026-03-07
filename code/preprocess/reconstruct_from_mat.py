"""Reconstruct CT volumes from projection-domain .mat files using CTorch FBP."""
import argparse
import glob
import os

import numpy as np
import scipy.io as sio
import tifffile as tiff
import torch
from CTorch.reconstructor.fbpreconstructor import FBPReconstructor

try:
    from .load_geometry import load_geometry_from_seq
except ImportError:
    from load_geometry import load_geometry_from_seq


def padding_mirror_u(arr_uva: np.ndarray, pad_len: int) -> np.ndarray:
    if pad_len == 0:
        return arr_uva
    u, v, a = arr_uva.shape
    out = np.zeros((u + 2 * pad_len, v, a), dtype=arr_uva.dtype)
    out[pad_len:pad_len + u, :, :] = arr_uva
    out[:pad_len, :, :] = arr_uva[pad_len - 1::-1, :, :]
    out[pad_len + u:, :, :] = arr_uva[:-pad_len - 1:-1, :, :]
    return out


def process_one(mat_path: str, seq_dir: str, out_dir: str, target_u: int):
    name = os.path.splitext(os.path.basename(mat_path))[0]
    print(f'\n=== {name} ===')

    mat = sio.loadmat(mat_path)
    if 'corr' not in mat:
        raise KeyError(f'{mat_path} is missing key: corr')

    corr_uva = mat['corr'].astype(np.float32)
    if corr_uva.ndim != 3:
        raise ValueError(f'Expected corr to have shape (U, V, A), got {corr_uva.shape}')

    pad = (target_u - corr_uva.shape[0]) // 2
    if pad < 0 or corr_uva.shape[0] + 2 * pad != target_u:
        raise ValueError(f'Cannot pad detector U from {corr_uva.shape[0]} to target_u={target_u}.')

    corr_padded = padding_mirror_u(corr_uva, pad)
    sino_avu = np.transpose(corr_padded, (2, 1, 0)).astype(np.float32)
    sino_t = torch.from_numpy(sino_avu).unsqueeze(0).unsqueeze(0).cuda()

    geo = load_geometry_from_seq(seq_dir, det_pad_u=pad)
    if int(geo.nu) != sino_t.shape[-1] or int(geo.nv) != sino_t.shape[-2]:
        raise ValueError(
            f'Geometry mismatch: geo(nv,nu)=({geo.nv},{geo.nu}) vs projection={tuple(sino_t.shape[-2:])}'
        )

    with torch.no_grad():
        reconstructor = FBPReconstructor(geo, algo='DD', parker=False).cuda()
        volume = reconstructor(sino_t)

    volume = volume.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    volume = np.flip(volume, axis=2)

    out_path = os.path.join(out_dir, f'reconstructed_{name}_f32.tiff')
    tiff.imwrite(out_path, volume, imagej=True, compression='deflate', metadata={'axes': 'ZYX'})
    print(f'Saved reconstruction: {out_path}')

    del sino_t
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Reconstruct CT volumes from .mat projection files')
    parser.add_argument('--data_dir', required=True, help='Directory containing sample*.mat files')
    parser.add_argument('--seq_dir', required=True, help='Directory containing geometry .seq files')
    parser.add_argument('--out_dir', required=True, help='Output directory for reconstructed TIFF volumes')
    parser.add_argument('--pattern', default='sample*.mat', help='Glob pattern for input .mat files')
    parser.add_argument('--target_u', type=int, default=2048, help='Target detector width after mirror padding')
    parser.add_argument('--gpu_idx', type=int, default=0, help='CUDA device index')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_idx)

    mat_paths = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))
    if not mat_paths:
        raise FileNotFoundError(f'No .mat files found under {args.data_dir} with pattern {args.pattern}')

    for mat_path in mat_paths:
        process_one(mat_path, args.seq_dir, args.out_dir, args.target_u)


if __name__ == '__main__':
    main()
