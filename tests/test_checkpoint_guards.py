"""Missing trained weights must fail before constructing SAM or reading files."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import test_stage2 as inference
import train_stage2


def forbid_expensive_work(*args, **kwargs):
    pytest.fail("SAM construction or checkpoint reading occurred before the missing-file check")


@pytest.mark.parametrize("stage1_exists", [False, True])
def test_inference_requires_both_trained_checkpoints_before_loading(
    tmp_path, monkeypatch, stage1_exists
):
    if stage1_exists:
        checkpoint = tmp_path / "stage1_unet" / "best.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"not a checkpoint; it must not be read")
    cfg = SimpleNamespace(
        OUTPUT_DIR=str(tmp_path), MODEL_TYPE="vit_b", SAM_CKPT="unused.pth"
    )
    monkeypatch.setitem(inference.sam_model_registry, "vit_b", forbid_expensive_work)
    monkeypatch.setattr(inference.torch, "load", forbid_expensive_work)
    expected_stage = "stage2_wfa" if stage1_exists else "stage1_unet"
    with pytest.raises(FileNotFoundError, match=expected_stage):
        inference.build_model(cfg, torch.device("cpu"))


def test_stage2_training_requires_trained_prompt_generator_first(tmp_path, monkeypatch):
    cfg = SimpleNamespace(
        OUTPUT_DIR=str(tmp_path), MODEL_TYPE="vit_b", SAM_CKPT="unused.pth"
    )
    monkeypatch.setitem(train_stage2.sam_model_registry, "vit_b", forbid_expensive_work)
    monkeypatch.setattr(train_stage2.torch, "load", forbid_expensive_work)
    with pytest.raises(FileNotFoundError, match="Stage 1 checkpoint"):
        train_stage2.build_model(cfg, torch.device("cpu"))
