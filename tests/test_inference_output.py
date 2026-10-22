"""Test evaluator threshold/unpadding independently of SAM's pretrained weights."""
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from config import Config
import test_stage2 as inference


class FixedLogitResult(nn.Module):
    """A deterministic model result fixture, not a substitute SAM implementation."""

    def __init__(self, probabilities):
        super().__init__()
        self.register_buffer("logits", torch.logit(torch.tensor(probabilities)))

    def forward(self, image, equalized, coarse):
        return self.logits, self.logits, torch.zeros_like(self.logits)


def test_evaluator_saves_unpadded_resized_grid_and_honors_configured_threshold(
    make_projection_case, tmp_path, monkeypatch
):
    image_cfg, name = make_projection_case()
    cfg = Config()
    for key, value in vars(image_cfg).items():
        setattr(cfg, key, value)
    cfg.TEST_NAMES = [name]
    cfg.OUTPUT_DIR = str(tmp_path / "evaluation")
    cfg.THRESHOLD = 0.8

    expected = np.zeros((6, 10), dtype=np.uint8)
    expected[:, 2:4] = 1
    expected[4:6, 8:10] = 1
    probabilities = np.full((1, 1, 10, 10), 0.7, dtype=np.float32)
    probabilities[0, 0, :6, :10][expected > 0] = 0.9
    probabilities[..., 6:, :] = 0.99
    fixed_result = FixedLogitResult(probabilities)
    monkeypatch.setattr(inference, "build_model", lambda cfg, device: fixed_result.to(device))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    metrics = inference.evaluate(cfg, save_predictions=True)

    assert metrics["dice"] == pytest.approx(1.0)
    assert metrics["hd95"] == pytest.approx(0.0)
    output = (Path(cfg.OUTPUT_DIR) / "eval_results" / "predictions"
              / name / "projection_view_000.png")
    pixels = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
    assert pixels is not None
    assert pixels.shape == (6, 10)
    np.testing.assert_array_equal(pixels, expected * 255)
