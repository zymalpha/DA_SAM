"""Regression cases for label scale, interpolation and unpadding metadata."""
import numpy as np
import pytest
import torch

from datasets.metal_dataset import (
    InferenceDataset,
    MetalArtifactDataset,
    Stage1Dataset,
)


@pytest.mark.parametrize("foreground", [1, 255])
@pytest.mark.parametrize(
    "dataset_type,mask_key",
    [(Stage1Dataset, "mask"), (MetalArtifactDataset, "mask_coarse"),
     (InferenceDataset, "mask")],
)
def test_thin_masks_keep_nearest_geometry_and_binary_scale(
    make_projection_case, foreground, dataset_type, mask_key
):
    cfg, name = make_projection_case(foreground)
    result = dataset_type([name], cfg)[0]

    # A 3x5 image becomes 6x10, then receives four zero rows of padding.
    # Nearest interpolation doubles each source pixel into a 2x2 block;
    # bilinear interpolation followed by >0 expands these thin structures.
    expected_mask = np.zeros((10, 10), dtype=np.float32)
    expected_mask[2:4, 4:6] = 1
    expected_target = np.zeros((10, 10), dtype=np.float32)
    expected_target[:6, 2:4] = 1
    expected_target[4:6, 8:10] = 1

    for key, expected in ((mask_key, expected_mask), ("gt", expected_target)):
        actual = result[key]
        if isinstance(actual, torch.Tensor):
            assert actual.dtype == torch.float32
            actual = actual.numpy()
        assert set(np.unique(actual)) == {0.0, 1.0}
        np.testing.assert_array_equal(np.squeeze(actual), expected)


def test_inference_records_resized_region_before_square_padding(make_projection_case):
    cfg, name = make_projection_case()
    result = InferenceDataset([name], cfg)[0]
    assert result["orig_hw"] == (3, 5)
    assert result["resized_hw"] == (6, 10)
    assert tuple(result["img_ori"].shape[-2:]) == (10, 10)
    height, width = result["resized_hw"]
    assert tuple(result["mask"][..., :height, :width].shape[-2:]) == (6, 10)
    assert torch.count_nonzero(result["mask"][..., height:, :]) == 0
