"""Check the numeric contract between labels, training targets and inference."""
import pytest
import torch
from thermalscout.model import ThermalScoutReal, CLASSES
from thermalscout.training import train_epoch, eval_epoch
from thermalscout.inference import nms
from thermalscout.checkpoint import load_detector


def test_nms_retains_strongest_overlapping_box():
    boxes = torch.tensor([[0.,0.,1.,1.],[.01,.01,.99,.99],[.8,.8,1.,1.]])
    assert nms(boxes, torch.tensor([.8,.9,.7])) == [1,2]


def test_validation_preserves_weights_and_batchnorm_statistics():
    model = ThermalScoutReal()
    loader = [(torch.rand(2,1,64,64), [[(0,.5,.5,.2,.2)]]*2, ['a','b'])]
    before = {k:v.clone() for k,v in model.state_dict().items()}
    eval_epoch(model, loader, 'cpu')
    assert not model.training
    assert all(torch.equal(v, before[k]) for k,v in model.state_dict().items())
    train_epoch(model, loader, torch.optim.AdamW(model.parameters()), 'cpu')
    assert model.training
    assert any(not torch.equal(v, before[k]) for k,v in model.state_dict().items())


def test_checkpoint_rejects_wrong_class_order_and_size(tmp_path):
    path = tmp_path/'weights.pt'
    ck = dict(model=ThermalScoutReal().state_dict(), classes=CLASSES, image_size=256)
    torch.save(ck, path)
    model, loaded = load_detector(path)
    assert not model.training and loaded['image_size'] == 256
    ck['classes'] = ('human', 'animal')
    torch.save(ck, path)
    with pytest.raises(ValueError, match='class order'):
        load_detector(path)
    ck.update(classes=CLASSES, image_size=255)
    torch.save(ck, path)
    with pytest.raises(ValueError, match='image_size'):
        load_detector(path)
