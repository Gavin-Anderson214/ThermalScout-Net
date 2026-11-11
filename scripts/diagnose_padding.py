"""Frozen-cache padding ablation and human-overlap diagnosis; development only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw
from thermalscout.data import BirdsaiDataset
from thermalscout.evaluate_detector import get_counts, xywh_to_xyxy, iou_xyxy


def content_rect(width, height, size):
    scale = min(size / width, size / height)
    nw, nh = max(1, round(width * scale)), max(1, round(height * scale))
    x, y = (size - nw) // 2, (size - nh) // 2
    return (x / size, y / size, (x + nw) / size, (y + nh) / size)


def inside(box, rect):
    x, y = (box[0]+box[2])/2, (box[1]+box[3])/2
    return rect[0] <= x < rect[2] and rect[1] <= y < rect[3]


def metrics(entries):
    classes = {name: get_counts(entries, c) for c, name in enumerate(('animal','human'))}
    tp, fp, gt = (sum(row[k] for row in classes.values()) for k in ('tp','fp','gt'))
    return dict(tp=tp, fp=fp, gt=gt, precision=tp/max(1,tp+fp), recall=tp/max(1,gt),
                f1=2*tp/max(1,tp+fp+gt), classes=classes)


def human_overlap(entries):
    # Per-label diagnostic only; candidates may overlap multiple labels.
    result = dict(human_labels=0, human_overlap_at_50=0, animal_only_overlap_at_50=0,
                  no_overlap_at_50=0, any_overlap_at_10=0)
    for entry in entries:
        preds = [p for p in entry['predictions'] if p[1]>=.25]
        for label in entry['ground_truth']:
            if int(label[0]) != 1: continue
            result['human_labels'] += 1
            box = xywh_to_xyxy(label)[1]
            overlaps = [(int(p[0]), iou_xyxy(p[2],box)) for p in preds]
            human = any(c==1 and v>=.5 for c,v in overlaps)
            animal = any(c==0 and v>=.5 for c,v in overlaps)
            result['human_overlap_at_50' if human else 'animal_only_overlap_at_50' if animal else 'no_overlap_at_50'] += 1
            result['any_overlap_at_10'] += int(any(v>=.1 for _,v in overlaps))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('data/processed/yolo_real'))
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    import torch
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    size=int(ck['image_size'])
    ds=BirdsaiDataset(a.data,'val',size,0,42,augment=False)
    entries=json.loads(a.cache.read_text(encoding='utf-8'))
    assert len(ds)==len(entries), 'Cache length differs from validation split'
    filtered=copy.deepcopy(entries)
    band_filtered=copy.deepcopy(entries)
    dataset_hash=hashlib.sha256()
    outside_labels=0
    # Validate names and transformed labels, fingerprint all current input bytes.
    for i,(image,label,_) in enumerate(ds.records):
        _, boxes, _ = ds[i]
        assert entries[i]['image']==image.name, 'Cache order/name mismatch'
        assert entries[i]['ground_truth']==[list(b) for b in boxes], 'Cache label mismatch'
        dataset_hash.update(image.name.encode()); dataset_hash.update(image.read_bytes()); dataset_hash.update(label.read_bytes())
        with Image.open(image) as im: rect=content_rect(*im.size,size)
        # Fixed four-pixel exclusion band inside the content boundary, a separate ablation.
        band_rect=(rect[0]+4/size,rect[1]+4/size,rect[2]-4/size,rect[3]-4/size)
        band_filtered[i]['predictions']=[pred for pred in entries[i]['predictions'] if inside(pred[2],band_rect)]
        outside_labels += sum(not inside(xywh_to_xyxy(b)[1],rect) for b in boxes)
        filtered[i]['predictions']=[pred for pred in entries[i]['predictions'] if inside(pred[2],rect)]
    before,after=metrics(entries),metrics(filtered)
    previous=json.loads((a.checkpoint.parent/'evaluation.json').read_text(encoding='utf-8'))
    for key in ('precision','recall'):
        assert abs(before[key]-previous[key])<1e-9, 'Saved evaluation mismatch'
    for name, counts in before['classes'].items():
        for key in ('tp','fp','gt'):
            assert counts[key]==previous['class_metrics'][name][key], 'Per-class baseline mismatch'
    summary=dict(scope='Development validation only; TestReal not used',
                 rule='Discard retained predicted boxes with centers outside the exact letterboxed content rectangle; no margin or tuned cutoff',
                 checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                 cache_sha256=hashlib.sha256(a.cache.read_bytes()).hexdigest(),
                 dataset_sha256=dataset_hash.hexdigest(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 images=len(ds),image_size=size, label_centers_outside_content=outside_labels,
                 baseline=before,padding_filter=after,
                 boundary_band_pixels=4,boundary_band_filter=metrics(band_filtered),
                 human_overlap_before=human_overlap(entries),human_overlap_after=human_overlap(filtered),
                 limitations=['Post-cap filtering cannot recover candidates removed by top-500 or 40-box caps.',
                              'Overlap categories are per label, not one-to-one matched recall or proof of causal class errors.',
                              'Cached predictions were generated at .001; all reported precision/recall use .25 confidence and .50 IoU.',
                              'No retraining, AP, or speed improvement is claimed. Training and inference defaults unchanged.'])
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # Two most human-heavy frames; fixed selection matches earlier examples.
    chosen=sorted(range(len(entries)),key=lambda i:-sum(int(b[0])==1 for b in entries[i]['ground_truth']))[:2]
    side=512; header=42
    canvas=Image.new('RGB',(side*4,(side+header)*len(chosen)),'#152033')
    pen=ImageDraw.Draw(canvas)
    for row,i in enumerate(chosen):
        tensor,labels,path=ds[i]
        base=Image.fromarray((tensor[0].numpy()*255).astype('uint8')).convert('RGB').resize((side,side))
        panels=[('Labels',[(int(b[0]),1.,xywh_to_xyxy(b)[1]) for b in labels]),
                ('Original >= .25',entries[i]['predictions']),('Padding filter >= .25',filtered[i]['predictions']),
                ('4px boundary band >= .25',band_filtered[i]['predictions'])]
        for col,(title,preds) in enumerate(panels):
            image=base.copy(); draw=ImageDraw.Draw(image)
            for c,score,box in preds:
                if score<.25: continue
                coords=[v*side for v in box]; color='#53c6ff' if c else '#a8ff78'
                draw.rectangle(coords,outline=color,width=2)
                draw.text((coords[0],max(0,coords[1]-12)),f'{"human" if c else "animal"} {score:.2f}',fill=color)
            x,y=col*side,row*(side+header)
            pen.text((x+5,y+3),title,fill='white');pen.text((x+5,y+20),Path(path).stem,fill='white')
            canvas.paste(image,(x,y+header))
    canvas.save(a.out/'human_padding_comparison.png')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
