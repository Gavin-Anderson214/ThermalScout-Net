"""Inspect a saved detector on development validation; never reads TestReal.

Caches predictions once, sweeps confidence thresholds, and draws error examples.
Run from the repository root with --run training_runs/<run_id>.
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch

from thermalscout.data import BirdsaiDataset
from thermalscout.evaluate_detector import get_counts, iou_xyxy, xywh_to_xyxy
from thermalscout.inference import infer
from thermalscout.model import ThermalScoutReal, CLASSES


def describe_errors(entry, threshold):
    """Match once per image; categorize unmatched predictions by box overlap.

    Categories describe annotation overlap, not confirmed real-world causes.
    Background-like means no label overlap at IoU >= .10, not proof of background.
    """
    truth = [xywh_to_xyxy(b) for b in entry['ground_truth']]
    used = set()
    result = dict(tp=0, duplicate=0, wrong_class=0, localization=0, background_like=0)
    for c, score, box in sorted(entry['predictions'], key=lambda p: -p[1]):
        if score < threshold:
            continue
        candidates = [(iou_xyxy(box, t), j) for j, (tc, t) in enumerate(truth)
                      if tc == c and j not in used]
        overlap, index = max(candidates, default=(0, -1))
        if index >= 0 and overlap >= .5:
            used.add(index)
            result['tp'] += 1
        elif any(tc == c and iou_xyxy(box, t) >= .5 for tc, t in truth):
            result['duplicate'] += 1
        elif any(tc != c and iou_xyxy(box, t) >= .5 for tc, t in truth):
            result['wrong_class'] += 1
        elif any(iou_xyxy(box, t) >= .1 for _, t in truth):
            result['localization'] += 1
        else:
            result['background_like'] += 1
    result['missed'] = len(truth) - len(used)
    return result


def draw_examples(dataset, entries, chosen, threshold, output):
    """Show one difficult frame per video: labels, baseline and filtered boxes."""
    side, header = 512, 52
    canvas = Image.new('RGB', (side*3, (side+header)*len(chosen)), '#101a29')
    for row, index in enumerate(chosen):
        tensor, labels, path = dataset[index]
        base = Image.fromarray((tensor[0].numpy()*255).astype('uint8')).convert('RGB')
        base = base.resize((side, side))
        truth = [(c, 1., xywh_to_xyxy(b)[1]) for b in labels for c in [int(b[0])]]
        panels = [('Labels', truth), ('Predictions >= 0.25', [p for p in entries[index]['predictions'] if p[1]>=.25]),
                  (f'Predictions >= {threshold:.2f}', [p for p in entries[index]['predictions'] if p[1]>=threshold])]
        for col, (title, boxes) in enumerate(panels):
            image = base.copy()
            pen = ImageDraw.Draw(image)
            for c, score, box in boxes:
                color = '#9ef6a7' if c == 0 else '#6dc7ff'
                coords = [v*side for v in box]
                pen.rectangle(coords, outline=color, width=2)
                pen.text((coords[0], max(0, coords[1]-12)), f'{CLASSES[c]} {score:.2f}', fill=color)
            x, y = col*side, row*(side+header)
            canvas.paste(image, (x,y+header))
            pen = ImageDraw.Draw(canvas)
            pen.text((x+8,y+5),title,fill='white')
            pen.text((x+8,y+24),Path(path).stem,fill='#a6bad3')
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/processed/yolo_real'))
    parser.add_argument('--out', type=Path, help='Separate analysis directory; must not already exist')
    args = parser.parse_args()
    checkpoint = args.run/'thermal_scout_real_weights.pt'
    output = args.out or args.run/'error_analysis'
    output.mkdir(parents=True, exist_ok=args.out is None)
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    model = ThermalScoutReal().eval()
    model.load_state_dict(saved['model'])
    dataset = BirdsaiDataset(args.data, 'val', int(saved['image_size']), 0, 42, augment=False)
    entries = []
    print(f'Analyzing all {len(dataset)} development validation frames on CPU.',flush=True)
    for index in range(len(dataset)):
        image, boxes, path = dataset[index]
        entries.append(dict(image=Path(path).name,ground_truth=boxes,
                            predictions=infer(model,image,'cpu',threshold=.001)))
        if (index+1)%50 == 0:
            print(f'  analyzed {index+1}/{len(dataset)}',flush=True)
    # Cache the exact capped/NMS-filtered predictions so the sweep is repeatable.
    (output/'predictions.json').write_text(json.dumps(entries),encoding='utf-8')
    thresholds = [.001,.05,.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.85,.9,.95,.975,.99,.995,.999,1.0]
    rows = []
    for threshold in thresholds:
        counts = [get_counts(entries,c,conf=threshold) for c in range(2)]
        for name, count in list(zip(CLASSES,counts))+[('all',dict(
                tp=sum(c['tp'] for c in counts),fp=sum(c['fp'] for c in counts),gt=sum(c['gt'] for c in counts)))]:
            tp, fp, gt = count['tp'],count['fp'],count['gt']
            precision, recall = tp/max(1,tp+fp),tp/max(1,gt)
            rows.append(dict(threshold=threshold, group=name,tp=tp,fp=fp,fn=gt-tp,gt=gt,
                             precision=precision,recall=recall,f1=2*tp/max(1,2*tp+fp+gt-tp)))
    baseline = next(r for r in rows if r['group']=='all' and r['threshold']==.25)
    previous = json.loads((args.run/'evaluation.json').read_text(encoding='utf-8'))
    assert abs(baseline['precision']-previous['precision'])<1e-9, 'Baseline precision mismatch'
    assert abs(baseline['recall']-previous['recall'])<1e-9, 'Baseline recall mismatch'
    for c, name in enumerate(CLASSES):
        current = get_counts(entries,c)
        for key in ('tp','fp','gt'):
            assert current[key] == previous['class_metrics'][name][key], f'{name} {key} mismatch'
    best = max((r for r in rows if r['group']=='all'),key=lambda r:(r['f1'],r['threshold']))
    with (output/'thresholds.csv').open('w',newline='',encoding='utf-8') as fp:
        writer = csv.DictWriter(fp,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Select animal filtering while keeping human threshold fixed at the baseline.
    # This guard preserves human TP on this cache, not on unseen data.
    human = next(r for r in rows if r['group']=='human' and r['threshold']==.25)
    policies = []
    for animal in (r for r in rows if r['group']=='animal'):
        tp, fp, gt = (animal[k]+human[k] for k in ('tp','fp','gt'))
        policies.append(dict(animal_threshold=animal['threshold'], human_threshold=.25,
                             tp=tp, fp=fp, gt=gt, precision=tp/max(1,tp+fp),
                             recall=tp/max(1,gt), f1=2*tp/max(1,tp+fp+gt)))
    policy = max(policies,key=lambda r:(r['f1'],r['animal_threshold']))
    with (output/'class_policies.csv').open('w',newline='',encoding='utf-8') as fp:
        writer = csv.DictWriter(fp,fieldnames=list(policies[0]));writer.writeheader();writer.writerows(policies)
    manifest = []
    for image, label, _ in dataset.records:
        manifest.append(dict(image=image.name, image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                             label_sha256=hashlib.sha256(label.read_bytes()).hexdigest()))
    (output/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    audit = []
    for ck in sorted(Path('training_runs').glob('*/thermal_scout_real_weights.pt')):
        state = torch.load(ck,map_location='cpu',weights_only=True)
        digest = hashlib.sha256()
        for key, value in sorted(state['model'].items()):
            digest.update(key.encode()); digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode()); digest.update(value.contiguous().numpy().tobytes())
        history = ck.parent/'loss_history.csv'
        audit.append(dict(run=ck.parent.name, checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),
                          tensor_sha256=digest.hexdigest(), epoch=state.get('epoch'),
                          objectness_loss=state.get('objectness_loss','not recorded'),
                          history_sha256=hashlib.sha256(history.read_bytes()).hexdigest() if history.exists() else None))
    (output/'checkpoint_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    errors = [describe_errors(e,.25) for e in entries]
    totals = {key:sum(e[key] for e in errors) for key in errors[0]}
    assert totals['tp']==baseline['tp']
    assert sum(totals[k] for k in ('duplicate','wrong_class','localization','background_like'))==baseline['fp']
    # Select one maximum-FP frame from each validation video; do not imply random sampling.
    by_video = {}
    for i, entry in enumerate(entries):
        video = entry['image'].split('__f')[0]
        fp = sum(errors[i][k] for k in ('duplicate','wrong_class','localization','background_like'))
        if video not in by_video or fp > by_video[video][0]:
            by_video[video]=(fp,i)
    chosen = [v[1] for _,v in sorted(by_video.items())]
    draw_examples(dataset,entries,chosen,best['threshold'],output/'validation_examples.png')
    # Include human-heavy frames explicitly; maximum-FP examples can all be animal-only.
    human_frames = [i for i,e in enumerate(entries) if any(int(b[0])==1 for b in e['ground_truth'])]
    human_chosen = sorted(human_frames, key=lambda i: -sum(int(b[0])==1 for b in entries[i]['ground_truth']))[:2]
    if human_chosen:
        draw_examples(dataset,entries,human_chosen,best['threshold'],output/'human_examples.png')
    summary = dict(run=args.run.name,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   checkpoint_epoch=saved['epoch'],images=len(entries),baseline=baseline,
                   per_class_baseline=[r for r in rows if r['threshold']==.25],
                   human_preserving_policy=policy,
                   policy_human_metrics=human,
                   reproducibility=dict(command=sys.argv, python=platform.python_version(), torch=torch.__version__,
                       numpy=np.__version__, device='cpu', data=str(args.data.resolve()),
                       dataset_manifest_sha256=hashlib.sha256((output/'dataset_manifest.json').read_bytes()).hexdigest(),
                       predictions_sha256=hashlib.sha256((output/'predictions.json').read_bytes()).hexdigest(),
                       source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           [Path(__file__), *Path('thermalscout').glob('*.py')]}),
                   validation_selected_best_micro_f1=best,baseline_error_categories=totals,
                   example_selection='Highest false-positive count frame in each validation video; ties use first filename',
                   example_images=[entries[i]['image'] for i in chosen],
                   human_example_selection='Two frames with most human labels; ties use first filename',
                   human_example_images=[entries[i]['image'] for i in human_chosen],
                   limitations=['Development validation threshold selection is optimistic; not official test performance.',
                                'Same 500 candidates, classwise NMS, and 40-detection cap as baseline; sweep only filters retained boxes.',
                                'FP categories are label-overlap heuristics, not visually confirmed causes.',
                                'No training or inference defaults changed.'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for name,color in [('all','#166534'),('animal','#2563eb'),('human','#9333ea')]:
        group = [r for r in rows if r['group']==name]
        axes[0].plot([r['recall'] for r in group],[r['precision'] for r in group],'.-',label=name,color=color)
        axes[1].plot([r['threshold'] for r in group],[r['f1'] for r in group],'.-',label=name,color=color)
    axes[0].set(xlabel='Recall',ylabel='Precision',title='Confidence tradeoff (IoU 0.50)',xlim=(0,1),ylim=(0,1))
    axes[1].set(xlabel='Confidence threshold',ylabel='F1',title='Validation threshold sweep',xlim=(0,1),ylim=(0,1))
    for ax in axes: ax.grid(alpha=.2);ax.legend()
    fig.savefig(output/'threshold_tradeoff.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)
    print('Saved analysis:',output,flush=True)


if __name__=='__main__':
    main()
