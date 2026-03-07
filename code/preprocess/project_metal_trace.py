"""Generate projection-domain training inputs from reconstructed volumes."""
import argparse
import glob
import os

import numpy as np
import scipy.io as sio
import tifffile as tiff
import torch
from CTorch.projector.projector_interface import Projector

try:
    from .load_geometry import load_geometry_from_seq
except ImportError:
    from load_geometry import load_geometry_from_seq


def sample_tag_from_path(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    return base.replace('reconstructed_', '').replace('_f32', '')


def export_projection_inputs(recon_tiff: str, mat_dir: str, projector, geo, out_dir: str, threshold: float):
    tag = sample_tag_from_path(recon_tiff)
    print(f'\n=== {tag} ===')

    vol_saved = tiff.imread(recon_tiff).astype(np.float32)
    if vol_saved.ndim != 3:
        raise ValueError(f'Expected reconstructed stack to have shape (Z, Y, X), got {vol_saved.shape}')

    vol = np.flip(vol_saved, axis=2)
    seg = (vol > threshold).astype(np.uint8)
    seg_u8 = (seg * 255).astype(np.uint8)
    seg_path = os.path.join(out_dir, f'seg_{tag}_u8.tiff')
    tiff.imwrite(seg_path, seg_u8, imagej=True, compression='deflate', metadata={'axes': 'ZYX'})
    print(f'Saved volume threshold segmentation: {seg_path}')

    mat_path = os.path.join(mat_dir, f'{tag}.mat')
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f'Missing .mat file for {tag}: {mat_path}')
    mat = sio.loadmat(mat_path)
    if 'corr' not in mat:
        raise KeyError(f'{mat_path} is missing key: corr')

    proj_img = np.transpose(mat['corr'].astype(np.float32), (2, 1, 0))
    expected_shape = (int(geo.nView), int(geo.nv), int(geo.nu))
    if proj_img.shape != expected_shape:
        raise ValueError(f'Projection shape mismatch for {tag}: {proj_img.shape} vs {expected_shape}')

    proj_img_path = os.path.join(out_dir, f'proj_img_{tag}_f32.tiff')
    tiff.imwrite(proj_img_path, proj_img, imagej=True, compression='deflate', metadata={'axes': 'ZYX'})
    print(f'Saved raw projections: {proj_img_path}')

    if 'metal_proj_trace' in mat:
        label = np.transpose((mat['metal_proj_trace'] > 0).astype(np.uint8), (2, 1, 0)) * 255
        label_path = os.path.join(out_dir, f'metal_trace_label_{tag}_u8.tiff')
        tiff.imwrite(label_path, label.astype(np.uint8), imagej=True, compression='deflate', metadata={'axes': 'ZYX'})
        print(f'Saved projection-domain labels: {label_path}')

    seg_t = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    with torch.no_grad():
        proj_seg = projector(seg_t)
    proj_seg = proj_seg.squeeze(0).squeeze(0).cpu().numpy()
    proj_metal = (proj_seg > 0).astype(np.uint8) * 255

    proj_metal_path = os.path.join(out_dir, f'proj_metal_{tag}_u8.tiff')
    tiff.imwrite(proj_metal_path, proj_metal.astype(np.uint8), imagej=True, compression='deflate', metadata={'axes': 'ZYX'})
    print(f'Saved forward-projected coarse masks: {proj_metal_path}')

    del seg_t
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Generate proj_img / proj_metal / labels from reconstructions')
    parser.add_argument('--recon_dir', required=True, help='Directory containing reconstructed_*.tiff volumes')
    parser.add_argument('--mat_dir', required=True, help='Directory containing source sample*.mat files')
    parser.add_argument('--seq_dir', required=True, help='Directory containing geometry .seq files')
    parser.add_argument('--out_dir', required=True, help='Output directory for projection-domain TIFF stacks')
    parser.add_argument('--pattern', default='reconstructed_sample*_f32.tiff', help='Glob pattern for reconstructed volumes')
    parser.add_argument('--threshold', type=float, default=0.08, help='Volume threshold used to create coarse metal masks')
    parser.add_argument('--gpu_idx', type=int, default=0, help='CUDA device index')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_idx)

    geo = load_geometry_from_seq(args.seq_dir, det_pad_u=0)
    projector = Projector(geo, projMode='proj', projAlgo='DD', projDirect='forward').cuda()

    recon_paths = sorted(glob.glob(os.path.join(args.recon_dir, args.pattern)))
    if not recon_paths:
        raise FileNotFoundError(f'No reconstructed volumes found under {args.recon_dir} with pattern {args.pattern}')

    for recon_tiff in recon_paths:
        export_projection_inputs(recon_tiff, args.mat_dir, projector, geo, args.out_dir, args.threshold)


if __name__ == '__main__':
    main()
