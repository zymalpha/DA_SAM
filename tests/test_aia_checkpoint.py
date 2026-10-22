"""Trainable modulation scale and strict, backward-compatible checkpoint loading."""
import io

import pytest
import torch
from torch import nn

from models.adapter import AIA


def test_modulation_scale_receives_gradients_and_survives_checkpoint_roundtrip():
    model = AIA(base_ch=8)
    assert isinstance(model.mod_scale, nn.Parameter)
    assert model.mod_scale.requires_grad
    assert model.mod_scale.item() == pytest.approx(0.5)
    assert "mod_scale" in dict(model.named_parameters())

    # The adapter head starts at zero, where the scale gradient is naturally
    # zero. A nonzero learned head exposes the scale's actual gradient path.
    with torch.no_grad():
        model.out_conv.bias.fill_(0.4)
    image = torch.ones(1, 1, 16, 16)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model(image, image).square().mean()
    loss.backward()
    assert model.mod_scale.grad is not None
    assert torch.isfinite(model.mod_scale.grad).all()
    assert model.mod_scale.grad.abs().item() > 0
    optimizer.step()
    assert model.mod_scale.item() != pytest.approx(0.5)

    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = AIA(base_ch=8)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    torch.testing.assert_close(restored.mod_scale, model.mod_scale)
    with torch.no_grad():
        torch.testing.assert_close(restored(image, image), model(image, image))


@pytest.mark.parametrize("nested", [False, True])
def test_legacy_checkpoint_uses_initialized_scale_and_warns_without_mutating_input(nested):
    def make_model(scale):
        aia = AIA(base_ch=8, mod_scale=scale)
        return nn.ModuleDict({"aia": aia}) if nested else aia

    source = make_model(0.25)
    legacy = source.state_dict()
    scale_key = "aia.mod_scale" if nested else "mod_scale"
    del legacy[scale_key]
    original_keys = set(legacy)
    restored = make_model(0.7)

    with pytest.warns(UserWarning, match="Legacy AIA checkpoint.*mod_scale"):
        result = restored.load_state_dict(legacy, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert set(legacy) == original_keys
    assert scale_key not in legacy
    restored_aia = restored["aia"] if nested else restored
    assert restored_aia.mod_scale.item() == pytest.approx(0.7)
    for key, value in legacy.items():
        torch.testing.assert_close(restored.state_dict()[key], value)


@pytest.mark.parametrize("problem", ["missing_weight", "unexpected_key", "scale_shape"])
def test_legacy_compatibility_does_not_disable_strict_checkpoint_validation(problem):
    model = AIA(base_ch=8)
    state = model.state_dict()
    if problem == "missing_weight":
        del state["out_conv.weight"]
    elif problem == "unexpected_key":
        state["not_a_model_parameter"] = torch.zeros(1)
    else:
        state["mod_scale"] = torch.zeros(2)
    with pytest.raises(RuntimeError):
        model.load_state_dict(state, strict=True)


@pytest.mark.parametrize("bad_scale", [
    torch.tensor([0.1, 0.2]),
    torch.tensor(float("nan")),
    torch.tensor(float("inf")),
    torch.tensor(0.1 + 0.2j),
    0.2,
])
@pytest.mark.parametrize("strict", [True, False])
def test_invalid_saved_scale_is_rejected_without_poisoning_parameter(bad_scale, strict):
    model = AIA(base_ch=8)
    state = model.state_dict()
    state["mod_scale"] = bad_scale
    with pytest.raises(RuntimeError, match="mod_scale"):
        model.load_state_dict(state, strict=strict)
    assert model.mod_scale.item() == pytest.approx(0.5)
    assert torch.isfinite(model.mod_scale)


@pytest.mark.parametrize("bad_scale", [float("nan"), float("inf"), -float("inf")])
def test_initial_modulation_scale_must_be_finite(bad_scale):
    with pytest.raises(ValueError, match="finite"):
        AIA(base_ch=8, mod_scale=bad_scale)
