import pytest

from thermalscout.data import tile_boxes, tile_positions


def test_windows_cover_edge_without_gaps():
    starts = tile_positions(698, 256)
    assert starts[0] == 0 and starts[-1] == 442
    assert all(next_start <= start + 256 for start, next_start in zip(starts, starts[1:]))


def test_tile_box_keeps_clipped_coordinates_and_rejects_small_fragment():
    # An object straddles the right edge of a 256-pixel tile.
    box = (1, 250/640, 128/512, 40/640, 20/512)
    labels = tile_boxes([box], 640, 512, 0, 0, 256)
    assert len(labels) == 1
    cls, x, y, w, h = labels[0]
    assert cls == 1
    assert x == pytest.approx(243/256)
    assert w == pytest.approx(26/256)
    assert y == pytest.approx(0.5)
    assert h == pytest.approx(20/256)
    assert tile_boxes([box], 640, 512, 256, 0, 256) == []


def test_tile_side_must_fit_source():
    with pytest.raises(ValueError):
        tile_positions(200, 256)
