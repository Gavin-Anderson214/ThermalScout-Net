import math
import pytest
import torch
from thermalscout.training import make_targets, criterion
from thermalscout.inference import infer
from thermalscout.evaluate_detector import get_counts, ap_101
from scripts.audit_bambi import species_class

def test_target_decoder_round_trip_both_classes():
    labels=[(0,.21,.34,.12,.18),(1,.72,.68,.08,.16)]
    obj,cls,loc,_=make_targets([labels],(1,7,16,16),'cpu')
    pred=torch.full((1,7,16,16),-20.)
    for c,x,y,w,h in labels:
        gx,gy=int(x*16),int(y*16)
        pred[0,0,gy,gx]=20
        pred[0,1:3,gy,gx]=torch.logit(loc[0,:2,gy,gx])
        pred[0,3:5,gy,gx]=loc[0,2:,gy,gx]
        pred[0,5+c,gy,gx]=20
    class Fixed(torch.nn.Module):
        def forward(self,x): return pred
    detections=infer(Fixed(),torch.zeros(1,64,64),'cpu',.25)
    entries=[dict(ground_truth=labels,predictions=detections)]
    for c in (0,1):
        assert get_counts(entries,c)['tp']==1
        assert ap_101(entries,c,.95)==pytest.approx(1)

def test_hard_negatives_have_finite_gradients_with_empty_image():
    pred=torch.zeros(1,7,16,16,requires_grad=True)
    loss,_=criterion(pred,[[]],objectness='hard-negative',background_weight=1)
    loss.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad[:,0].abs().sum()>0

def test_hard_mining_emphasizes_sparse_false_alarms():
    pred=torch.full((1,7,16,16),-8.)
    pred[0,0,0,0]=8
    base,_=criterion(pred,[[]])
    hard,_=criterion(pred,[[]],objectness='hard-negative')
    assert hard>base*4

@pytest.mark.parametrize('weight',[0,-1,float('nan'),float('inf')])
def test_invalid_background_weight(weight):
    with pytest.raises(ValueError): criterion(torch.zeros(1,7,2,2),[[]],background_weight=weight)

def test_species_mapping():
    assert species_class('Homo sapiens (Human)')==1
    assert species_class('Sus scrofa × Sus domesticus (Hybrid pig)')==0
    assert species_class('Cervus elaphus (Red deer)')==0
    assert species_class('Unknown') is None
    assert species_class('No-animal') is None

def test_local_peaks_and_output_cap():
    pred=torch.full((1,7,4,4),-10.)
    pred[:,1:5]=0
    pred[:,5]=10
    pred[0,0,1,1]=6
    pred[0,0,1,2]=5
    pred[0,0,3,3]=4
    class Fixed(torch.nn.Module):
        def forward(self,x): return pred
    image=torch.zeros(1,16,16)
    assert len(infer(Fixed(),image,'cpu',.25,local_peaks=True))==2
    assert len(infer(Fixed(),image,'cpu',.25,max_detections=1))==1
    with pytest.raises(ValueError):infer(Fixed(),image,'cpu',max_candidates=0)
