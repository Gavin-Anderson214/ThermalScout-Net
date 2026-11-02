"""Lightweight checks for fundamental code behavior; these are not accuracy tests."""
import math
from pathlib import Path
import pytest
import torch
from thermalscout.model import ThermalScoutReal
from thermalscout.training import criterion
from thermalscout.data import read_boxes
from thermalscout.evaluate_detector import iou_xyxy
from thermalscout.inference import infer


def test_model_returns_seven_prediction_channels():
    # One objectness, four location and two class numbers per grid cell.
    m = ThermalScoutReal().eval()
    with torch.no_grad():
        result = m(torch.zeros(1, 1, 64, 64))
    assert result.shape == (1, 7, 16, 16)


def test_loss_is_finite_even_with_zero_objects():
    m = ThermalScoutReal().eval()
    with torch.no_grad():
        pred = m(torch.zeros(1, 1, 64, 64))
    assert math.isfinite(float(criterion(pred, [[]])[0]))
    assert math.isfinite(float(criterion(pred, [[(0, 0.5, 0.5, 0.1, 0.1)]])[0]))


def test_read_yolo_labels(tmp_path):
    file = tmp_path / "image.txt"
    file.write_text("0 0.5 0.5 0.2 0.2\n1 0.25 0.75 0.1 0.1\n")
    assert len(read_boxes(file)) == 2
    file.write_text("3 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError): read_boxes(file)


def test_overlap():
    assert iou_xyxy([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.)
    assert iou_xyxy([0, 0, .1, .1], [.9, .9, 1, 1]) == 0.


def test_inference_produces_valid_boxes():
    model = ThermalScoutReal().eval()
    dets = infer(model, torch.zeros(1, 64, 64), "cpu", threshold=0.01)
    assert all(c in (0, 1) and 0. <= s <= 1. and len(box) == 4 for c,s,box in dets)


def test_focal_background_weights_hard_negatives():
    # Two all-background cells: the high-confidence mistake should dominate.
    pred = torch.zeros(1, 7, 1, 2, requires_grad=True)
    with torch.no_grad():
        pred[0, 0, 0, 0] = -4.0
        pred[0, 0, 0, 1] = 4.0
    focal, _ = criterion(pred, [[]], objectness="focal")
    focal.backward()
    assert torch.isfinite(focal)
    assert abs(pred.grad[0, 0, 0, 1]) > 100 * abs(pred.grad[0, 0, 0, 0])
    assert math.isfinite(float(criterion(torch.zeros(1, 7, 2, 2), [[(1, .5, .5, .2, .2)]], objectness="focal")[0]))
