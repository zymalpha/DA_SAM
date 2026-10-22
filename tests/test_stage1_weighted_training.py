"""Exercise both Stage I loops with a real tiny CPU U-Net and real PNG data."""
import csv
from pathlib import Path

import pytest
import torch

from config import Config
from train_stage1 import train


def test_stage1_configured_loss_weights_control_training_and_validation(
    make_projection_case, tmp_path, monkeypatch
):
    image_cfg, name = make_projection_case()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    results = {}
    saved_weights = []
    for run_name, bce_weight, dice_weight in (
        ("bce_only", 1.0, 0.0),
        ("dice_only", 0.0, 1.0),
        ("weighted", 0.8, 2.0),
    ):
        cfg = Config()
        for key, value in vars(image_cfg).items():
            setattr(cfg, key, value)
        cfg.IMG_SIZE = 16
        cfg.TRAIN_NAMES = [name]
        cfg.VAL_NAMES = [name]
        cfg.OUTPUT_DIR = str(tmp_path / run_name)
        cfg.UNET_BASE_CH = 8
        cfg.UNET_DEPTH = 1
        cfg.STAGE1_BATCH_SIZE = 1
        cfg.STAGE1_EPOCHS = 1
        cfg.NUM_WORKERS = 0
        cfg.LR = 0.0
        cfg.WEIGHT_DECAY = 0.0
        cfg.LAMBDA_BCE = bce_weight
        cfg.LAMBDA_DICE = dice_weight
        train(cfg)

        directory = Path(cfg.OUTPUT_DIR) / "stage1_unet"
        with (directory / "log.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        results[run_name] = rows[0]
        checkpoint = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        saved_weights.append(checkpoint["unet"])

    # Zero learning rate keeps identical initial networks across all runs.
    # The observed CSV losses must change with each independently configured
    # component, in BOTH loops, instead of silently using an unweighted sum.
    for other in saved_weights[1:]:
        for key, value in saved_weights[0].items():
            torch.testing.assert_close(other[key], value, rtol=0, atol=0)
    for column in ("train_loss", "val_loss"):
        bce = float(results["bce_only"][column])
        dice = float(results["dice_only"][column])
        weighted = float(results["weighted"][column])
        assert bce > 0
        assert dice > 0
        assert weighted == pytest.approx(0.8 * bce + 2.0 * dice, abs=3e-6)
        assert weighted != pytest.approx(bce + dice, abs=1e-4)
