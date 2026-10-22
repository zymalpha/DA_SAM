"""Small, real image fixtures; no SAM checkpoint or GPU is required."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def make_projection_case(tmp_path):
    def make_case(foreground=255):
        cfg = SimpleNamespace(
            IMG_SIZE=10,
            DIR_PROJ=str(tmp_path / "projections"),
            DIR_EQ=str(tmp_path / "equalized"),
            DIR_PROJ_METAL=str(tmp_path / "coarse"),
            DIR_LABEL=str(tmp_path / "labels"),
        )
        name = "synthetic_patient"
        filename = "projection_view_000.png"
        raw = np.arange(15, dtype=np.uint8).reshape(3, 5) * 17
        coarse = np.zeros((3, 5), dtype=np.uint8)
        coarse[1, 2] = foreground
        target = np.zeros((3, 5), dtype=np.uint8)
        target[:, 1] = foreground
        target[2, 4] = foreground
        for folder, pixels in (
            (cfg.DIR_PROJ, raw),
            (cfg.DIR_EQ, 255 - raw),
            (cfg.DIR_PROJ_METAL, coarse),
            (cfg.DIR_LABEL, target),
        ):
            patient_dir = Path(folder) / name
            patient_dir.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(patient_dir / filename), pixels)
        return cfg, name

    return make_case
