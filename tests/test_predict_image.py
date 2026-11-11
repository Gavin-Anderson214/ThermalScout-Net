"""Prevent training/inference drift and padding mistakes on real camera shapes."""
import numpy as np
from PIL import Image
import pytest
import torch

from thermalscout.data import BirdsaiDataset
from thermalscout.predict_image import prepare_image, original_boxes


@pytest.mark.parametrize('shape', [(641, 479), (479, 641), (64, 64)])
def test_preprocessing_matches_training_and_boxes_round_trip(tmp_path, shape):
    images = tmp_path / 'images' / 'val'
    labels = tmp_path / 'labels' / 'val'
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    width, height = shape
    rng = np.random.default_rng(42)
    image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
    image.save(images / 'frame.png')
    (labels / 'frame.txt').write_text('1 0.5 0.5 0.4 0.2\n')
    expected, boxes, _ = BirdsaiDataset(tmp_path, 'val', 256, 0, 42)[0]
    actual, geometry = prepare_image(image, 256)
    assert torch.equal(actual, expected)
    c, x, y, w, h = boxes[0]
    result = original_boxes([(c, 0.9, [x-w/2, y-h/2, x+w/2, y+h/2])], geometry)
    assert result[0]['box_xyxy_pixels'] == pytest.approx([width*.3, height*.4, width*.7, height*.6])


def test_padding_only_box_removed_and_crossing_box_clipped():
    _, geometry = prepare_image(Image.new('L', (640, 320)), 256)
    results = original_boxes([(0, .9, [0, 0, 1, .2]),
                              (1, .8, [0, 0, 1, 1])], geometry)
    assert len(results) == 1
    assert results[0]['box_xyxy_pixels'] == [0, 0, 640, 320]


@pytest.mark.parametrize('size', [0, -8, 255])
def test_invalid_model_size_rejected(size):
    with pytest.raises(ValueError):
        prepare_image(Image.new('L', (64, 64)), size)
