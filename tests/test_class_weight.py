import pytest
import torch

from thermalscout.training import criterion


def test_human_weight_default_preserves_baseline_and_changes_mixed_gradients():
    # Give one cell an animal label and the other a human label. Equal starting
    # scores let us isolate what changes when the human class gets more weight.
    boxes = [[(0, .25, .5, .1, .1), (1, .75, .5, .1, .1)]]
    predictions = torch.zeros(1, 7, 1, 2, requires_grad=True)
    default_loss, _ = criterion(predictions, boxes)
    baseline_loss, _ = criterion(predictions, boxes, human_weight=2.)
    assert torch.equal(default_loss, baseline_loss)

    # Channel 6 is the human class score. Compare its gradient at the human
    # cell, clearing the first gradient so the two measurements stay separate.
    human_score_index = (0, 6, 0, 1)
    baseline_loss.backward()
    original_gradient = predictions.grad[human_score_index].abs().item()
    predictions.grad.zero_()

    increased_loss, _ = criterion(predictions, boxes, human_weight=4.)
    increased_loss.backward()
    assert predictions.grad[human_score_index].abs().item() > original_gradient

    for invalid_weight in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            criterion(predictions, boxes, human_weight=invalid_weight)
