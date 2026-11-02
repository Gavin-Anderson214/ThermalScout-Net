import pytest
import torch
from thermalscout.training import criterion


def test_human_weight_default_preserves_baseline_and_changes_mixed_gradients():
    boxes=[[(0,.25,.5,.1,.1),(1,.75,.5,.1,.1)]]
    pred=torch.zeros(1,7,1,2,requires_grad=True)
    default,_=criterion(pred,boxes)
    baseline,_=criterion(pred,boxes,human_weight=2.)
    assert torch.equal(default,baseline)
    baseline.backward()
    before=pred.grad[0,6,0,1].abs().item()
    pred.grad.zero_()
    increased,_=criterion(pred,boxes,human_weight=4.)
    increased.backward()
    assert pred.grad[0,6,0,1].abs().item()>before
    for bad in (0,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError): criterion(pred,boxes,human_weight=bad)
