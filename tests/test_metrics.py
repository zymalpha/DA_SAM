"""Analytically known geometry catches foreground-versus-boundary EDT errors."""
import numpy as np
import pytest

from utils.metrics import compute_hd95, get_full_metrics


def test_hd95_detects_an_interior_hole_even_when_all_foreground_overlaps():
    target = np.zeros((21, 21), dtype=np.uint8)
    target[3:18, 3:18] = 1
    prediction = target.copy()
    prediction[9:12, 9:12] = 0

    # The new inner contour is five pixels from the outer reference contour.
    # Measuring distance to reference foreground instead of its contour gives 0.
    assert compute_hd95(prediction, target) == pytest.approx(5.0)
    assert compute_hd95(target, prediction) == pytest.approx(5.0)
    assert get_full_metrics(prediction, target)[2] == pytest.approx(5.0)


def test_hd95_uses_euclidean_pixel_distance():
    target = np.zeros((10, 10), dtype=np.uint8)
    prediction = np.zeros_like(target)
    target[2, 2] = 1
    prediction[5, 6] = 1
    assert compute_hd95(prediction, target) == pytest.approx(5.0)


def test_hd95_keeps_pooled_bidirectional_percentile_convention():
    target = np.zeros((5, 7), dtype=np.uint8)
    prediction = np.zeros_like(target)
    target[2, 1] = 1
    prediction[2, 1:6] = 1
    # Directed distances are [0, 1, 2, 3, 4] and [0]. Their pooled 95th
    # percentile is 3.75; max(directional percentiles) would instead be 3.8.
    assert compute_hd95(prediction, target) == pytest.approx(3.75)


@pytest.mark.parametrize("pred_empty,gt_empty,expected", [
    (True, True, 0.0), (True, False, 50.0), (False, True, 50.0),
])
def test_empty_mask_conventions_are_preserved(pred_empty, gt_empty, expected):
    prediction = np.zeros((5, 5), dtype=np.uint8)
    target = prediction.copy()
    if not pred_empty:
        prediction[2, 2] = 1
    if not gt_empty:
        target[2, 2] = 1
    assert compute_hd95(prediction, target) == expected
    assert get_full_metrics(prediction, target)[2] == expected


def test_overlap_metrics_treat_positive_mask_encodings_as_foreground():
    prediction = np.zeros((6, 6), dtype=np.uint8)
    target = np.zeros_like(prediction)
    prediction[1:4, 1:4] = 1
    target[2:5, 1:4] = 1
    expected = (2 / 3, 1 / 2, 1.0, 2 / 3, 2 / 3)
    assert get_full_metrics(prediction, target) == pytest.approx(expected)
    assert get_full_metrics(prediction * 255, target * 255) == pytest.approx(expected)


@pytest.mark.parametrize("metric", [compute_hd95, get_full_metrics])
@pytest.mark.parametrize("prediction,target", [
    (np.zeros((2, 3)), np.zeros((3, 2))),
    (np.zeros(3), np.zeros(3)),
    (np.empty((0, 3)), np.empty((0, 3))),
    (np.array([[np.nan]]), np.array([[0.0]])),
    (np.array([[1.0]]), np.array([[np.inf]])),
])
def test_metrics_reject_undefined_or_incompatible_inputs(metric, prediction, target):
    with pytest.raises(ValueError):
        metric(prediction, target)
