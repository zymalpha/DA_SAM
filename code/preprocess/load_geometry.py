"""Geometry loader used by CTorch-based preprocessing scripts."""
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from CTorch.utils.geometry import CircGeom3D


@dataclass
class GeometryConfig:
    detector_v: int = 1024
    detector_u: int = 1024
    detector_spacing_v: float = 0.3
    detector_spacing_u: float = 0.3
    voxel_x: int = 400
    voxel_y: int = 400
    voxel_z: int = 400
    voxel_spacing_x: float = 0.5
    voxel_spacing_y: float = 0.5
    voxel_spacing_z: float = 0.5


def _read_seq(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        return np.fromfile(f, dtype=np.float32)


def load_geometry_from_seq(seq_dir: str, det_pad_u: int = 0, cfg: Optional[GeometryConfig] = None):
    cfg = GeometryConfig() if cfg is None else cfg

    angles = _read_seq(os.path.join(seq_dir, 'angle.seq'))
    angles = -angles[::-1] + math.pi * 3.25 / 4 + math.pi / 2

    sys_m = _read_seq(os.path.join(seq_dir, 'SysM.seq'))

    sdd_candidates = ['sdd.seq', 'SDD.seq', 'SOD.seq']
    sdd = None
    for candidate in sdd_candidates:
        path = os.path.join(seq_dir, candidate)
        if os.path.exists(path):
            sdd = _read_seq(path)
            break
    if sdd is None:
        raise FileNotFoundError('Could not find sdd.seq, SDD.seq, or SOD.seq in geometry directory.')

    sod = sdd / sys_m

    det_x = _read_seq(os.path.join(seq_dir, 'DetX.seq'))
    det_y = _read_seq(os.path.join(seq_dir, 'DetY.seq'))
    p_y = _read_seq(os.path.join(seq_dir, 'pY.seq')) - 1

    z_src = -(p_y - det_y) * cfg.detector_spacing_v / sys_m
    padded_u = int(cfg.detector_u + 2 * det_pad_u)

    v_offset = -(det_y - cfg.detector_v / 2 + 0.5) * cfg.detector_spacing_v + z_src
    u_offset = -((padded_u / 2 - 0.5) - (det_x + det_pad_u)) * cfg.detector_spacing_u

    geo = CircGeom3D(
        nx=cfg.voxel_x,
        ny=cfg.voxel_y,
        nz=cfg.voxel_z,
        dx=cfg.voxel_spacing_x,
        dy=cfg.voxel_spacing_y,
        dz=cfg.voxel_spacing_z,
        nView=len(angles),
        nu=padded_u,
        nv=cfg.detector_v,
        du=cfg.detector_spacing_u,
        dv=cfg.detector_spacing_v,
        detType='flat',
        SAD=[float(s) for s in sod],
        SDD=[float(s) for s in sdd],
        xOfst=[0.0],
        yOfst=[0.0],
        zOfst=[0.0],
        uOfst=u_offset.tolist(),
        vOfst=v_offset.tolist(),
        xSrc=[0.0],
        zSrc=z_src.tolist(),
        viewAngles=angles,
        fixed=False,
    )
    return geo
