"""Evaluate a trained custom ThermalScout-Net on prepared validation imagery.

Reports detection precision/recall at 0.25 confidence and IoU 0.50,
per-class AP at IoU 0.50, and mAP averaged over IoU 0.50...0.95.
This is a DEVELOPMENT evaluation, not an untouched official-test score.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import ThermalScoutReal, CLASSES
from .data import BirdsaiDataset, collate
from .inference import infer


def iou_xyxy(a, b):
    """Calculate overlap between two boxes as a fraction from 0 to 1."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0., x2-x1) * max(0., y2-y1)
    area_a = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return inter / max(1.e-9, area_a+area_b-inter)


def xywh_to_xyxy(box):
    """Convert a label from center/size to top-left and bottom-right corners."""
    cls,x,y,w,h = box
    return int(cls), [max(0,x-w/2),max(0,y-h/2),min(1,x+w/2),min(1,y+h/2)]


def get_counts(entries, class_id, iou=0.5, conf=.25):
    """Count correct detections and mistakes for precision and recall."""
    hits, false_positives, ground_truths = 0,0,0
    for entry in entries:
        truth=[xywh_to_xyxy(b)[1] for b in entry['ground_truth'] if int(b[0])==class_id]
        preds=sorted([p for p in entry['predictions'] if p[0]==class_id and p[1]>=conf],key=lambda p:-p[1])
        ground_truths += len(truth)
        used=set()
        for _,_,box in preds:
            choices=[(iou_xyxy(box,t),j) for j,t in enumerate(truth) if j not in used]
            score,j=max(choices,default=(0.,-1))
            if j>=0 and score>=iou: used.add(j); hits+=1
            else: false_positives+=1
    return dict(tp=hits,fp=false_positives,gt=ground_truths,
                precision=hits/max(1,hits+false_positives),
                recall=hits/max(1,ground_truths))


def ap_101(entries,class_id,threshold):
    """Calculate average precision from the full ranked prediction list."""
    truths={}
    all_preds=[]
    gt_count=0
    for img_id,entry in enumerate(entries):
        t=[xywh_to_xyxy(b)[1] for b in entry['ground_truth'] if int(b[0])==class_id]
        truths[img_id]=t;gt_count+=len(t)
        for cl,score,box in entry['predictions']:
            if cl==class_id: all_preds.append((float(score),img_id,box))
    if gt_count==0:
        return None
    all_preds.sort(key=lambda p:-p[0])
    matched={k:set() for k in truths}
    tp=[];fp=[]
    for _,img_id,box in all_preds:
        options=[(iou_xyxy(box,v),j) for j,v in enumerate(truths[img_id]) if j not in matched[img_id]]
        best,j=max(options,default=(0.,-1))
        ok=j>=0 and best>=threshold
        if ok:matched[img_id].add(j)
        tp.append(int(ok));fp.append(int(not ok))
    if not tp:return 0.0
    ctp=np.cumsum(tp);cfp=np.cumsum(fp)
    recall=ctp/gt_count
    precision=ctp/np.maximum(1,ctp+cfp)
    return float(np.mean([np.max(precision[recall>=r],initial=0.) for r in np.linspace(0.,1.,101)]))


def main():
    """Load the trained model, measure validation metrics, and save JSON."""
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--limit',type=int,default=0,help='0 = entire development validation subset')
    args=p.parse_args()
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    if device=='cuda' and not torch.cuda.is_available():p.error('CUDA not available')
    if not args.checkpoint.exists():p.error('Training checkpoint not found')
    ck=torch.load(args.checkpoint,map_location=device,weights_only=True)
    size=int(ck.get('image_size',256))
    model=ThermalScoutReal().to(device)
    model.load_state_dict(ck['model']);model.eval()
    ds=BirdsaiDataset(args.data,'val',size,args.limit,42,augment=False)
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate)
    entries=[]; secs=0.
    print(f'Evaluating {len(ds)} video-held-out DEVELOPMENT validation frames on {device}...')
    for n,(img,boxes,_) in enumerate(loader,1):
        tensor=img[0]
        if device=='cuda':torch.cuda.synchronize()
        start=time.perf_counter()
        dets=infer(model,tensor,device,threshold=0.001)
        if device=='cuda':torch.cuda.synchronize()
        secs+=time.perf_counter()-start
        entries.append({'ground_truth':boxes[0], 'predictions':dets})
        if n%50==0:print('  evaluated',n,'/',len(ds),flush=True)
    iou_steps=np.arange(0.5,1.0,0.05)
    class_aps={}
    count_by_class={}
    for c,name in enumerate(CLASSES):
        counts=get_counts(entries,c)
        aps=[ap_101(entries,c,float(th)) for th in iou_steps]
        counts['ap50']=aps[0]
        counts['map5095']=float(np.mean([x for x in aps if x is not None])) if any(x is not None for x in aps) else None
        class_aps[name]=counts
        count_by_class[c]=counts
    map50=float(np.mean([x['ap50'] for x in class_aps.values() if x['ap50'] is not None]))
    map5095=float(np.mean([x['map5095'] for x in class_aps.values() if x['map5095'] is not None]))
    total_tp=sum(x['tp'] for x in count_by_class.values())
    total_fp=sum(x['fp'] for x in count_by_class.values())
    total_gt=sum(x['gt'] for x in count_by_class.values())
    out={
        'model':'ThermalScout-Net (real/scratch)',
        'training':'random initialization',
        'split':'validation',
        'scope':'DEVELOPMENT ONLY - official TestReal was not used',
        'evaluation_images':len(ds),'image_size':size,'iou_for_precision_recall':0.5,
        'confidence_for_precision_recall':0.25,
        'ap_method':'101-point interpolated per class; IoU 0.50..0.95 (step 0.05)',
        'precision':total_tp/max(1,total_tp+total_fp),
        'recall':total_tp/max(1,total_gt),
        'map50':map50,'map5095':map5095,
        'fps':len(ds)/max(1.e-9,secs),
        'fps_measurement':'forward pass + CPU/GPU postprocessing; no file-I/O or dataset-loading time',
        'class_metrics':class_aps,
        'notes':'Small prototype; one box per stride-4 cell, predictions capped at 40/image.'
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('Saved metrics:',args.out)
    print('micro precision:',round(out['precision'],4),'micro recall:',round(out['recall'],4),
          'mAP50:',round(map50,4),'mAP50:95:',round(map5095,4),'fps:',round(out['fps'],2))
    print('CAUTION: development validation scores, not final test accuracy.')

if __name__=='__main__':main()
