"""Reproducible prepared/source audit and real-image memorization gate.

Run as python -m scripts.verify_pipeline {audit,overfit} --out NEW_DIRECTORY.
All datasets and historical checkpoints remain read-only.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import random
import platform
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from thermalscout.data import BirdsaiDataset, collate
from thermalscout.model import CLASSES, ThermalScoutReal
from thermalscout.training import train_epoch, eval_epoch, make_targets
from thermalscout.inference import infer
from thermalscout.predict_image import prepare_image
from thermalscout.evaluate_detector import get_counts, ap_101
from scripts.birdsai_prepare import read_annotations, clip_box


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def metrics(entries):
    classes = {}
    for c, name in enumerate(CLASSES):
        classes[name] = {**get_counts(entries, c), 'ap50': ap_101(entries, c, .5)}
        classes[name]['fn'] = classes[name]['gt'] - classes[name]['tp']
    return {'map50': float(np.mean([v['ap50'] for v in classes.values() if v['ap50'] is not None])),
            'class_metrics': classes}


@torch.no_grad()
def predict(model, samples, device):
    return [dict(path=path, ground_truth=boxes,
                 predictions=infer(model, image, device, threshold=.001))
            for image, boxes, path in samples]


def audit(args):
    root = args.data
    manifest = list(csv.DictReader((root / 'manifest.csv').open(encoding='utf-8')))
    split_info = json.loads((root / 'split_videos.json').read_text())
    hashes = defaultdict(list)
    pixel_hashes = defaultdict(list)
    result = {'classes': list(CLASSES), 'format': 'class normalized_center_x center_y width height',
              'split_metadata': split_info, 'issues': [], 'splits': {}}
    source_cache = {}
    if args.source:
        total = Counter()
        image_count = annotation_duplicates = 0
        for label in sorted((args.source/'annotations').glob('*.csv')):
            annotations, counts = read_annotations(label)
            total.update(counts)
            source_cache[label.stem] = annotations
            annotation_duplicates += sum(len(rows)-len(set(rows)) for rows in annotations.values())
            folder = args.source/'images'/label.stem
            if folder.is_dir():
                image_count += sum(p.suffix.lower() in {'.jpg','.jpeg','.png'} for p in folder.iterdir())
        result['original_TrainReal'] = dict(images=image_count, sequences=len(source_cache),
            annotation_counts=dict(total), duplicate_annotation_rows=annotation_duplicates)
    for split in ('train', 'val'):
        ds = BirdsaiDataset(root, split, args.size, 0, 42)
        videos, humans = Counter(), Counter()
        duplicates = bounds = mismatched = preprocess_mismatches = collisions = 0
        source_checked = source_mismatched = 0
        rows = {f"{r['video']}__f{int(r['original_frame_id']):07d}": r
                for r in manifest if r['split'] == split}
        for i, (path, label, boxes) in enumerate(ds.records):
            row = rows.get(path.stem)
            if row is None:
                result['issues'].append(f'No manifest row: {split}/{path.name}')
                continue
            video = row['video']
            videos[video] += 1
            humans[video] += sum(b[0] == 1 for b in boxes)
            duplicates += len(boxes) - len(set(boxes))
            bounds += sum(x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6
                          for c,x,y,w,h in boxes)
            mismatched += int(row['valid_box_count']) != len(boxes)
            hashes[digest(path)].append(f'{split}/{path.name}')
            with Image.open(path) as im:
                gray = im.convert('L')
                pixel_hashes[hashlib.sha256(gray.tobytes()).hexdigest()].append(f'{split}/{path.name}')
                tensor, geometry = prepare_image(im, args.size)
                width, height = im.size
            loaded, adjusted, _ = ds[i]
            preprocess_mismatches += not torch.equal(tensor, loaded)
            collisions += make_targets([adjusted], (1,7,args.size//4,args.size//4), 'cpu')[3]
            if args.source:
                if video not in source_cache:
                    source_cache[video] = read_annotations(args.source/'annotations'/f'{video}.csv')[0]
                expected = []
                for ann in source_cache[video].get(int(row['original_frame_id']), []):
                    clipped = clip_box(ann, width, height)
                    if clipped:
                        x1,y1,x2,y2 = clipped
                        expected.append((ann.cls,(x1+x2)/(2*width),(y1+y2)/(2*height),
                                         (x2-x1)/width,(y2-y1)/height))
                source_checked += 1
                source_image = args.source/'images'/video/row['original_image_name']
                source_mismatched += (len(expected) != len(boxes) or
                                     (bool(boxes) and not np.allclose(expected, boxes, atol=5.1e-7, rtol=0)) or
                                     digest(source_image) != digest(path))
        orphan = [p.name for p in (root/'labels'/split).glob('*.txt')
                  if p.stem not in {r[0].stem for r in ds.records}]
        result['splits'][split] = dict(images=len(ds), label_files=len(list((root/'labels'/split).glob('*.txt'))),
            boxes=dict(zip(CLASSES, ds.label_counts)), empty_images=ds.num_empty,
            videos=dict(videos), human_boxes_by_video={k:v for k,v in humans.items() if v},
            duplicate_annotations=duplicates, out_of_bounds=bounds, manifest_count_mismatches=mismatched,
            preprocessing_mismatches=preprocess_mismatches, same_cell_collisions=collisions,
            orphan_labels=orphan, source_frames_checked=source_checked, source_mismatches=source_mismatched)
        print(split, result['splits'][split], flush=True)
    train_videos = set(result['splits']['train']['videos'])
    val_videos = set(result['splits']['val']['videos'])
    result['shared_video_ids'] = sorted(train_videos & val_videos)
    result['shared_filename_prefixes'] = sorted({v.split('_')[0] for v in train_videos} &
                                                {v.split('_')[0] for v in val_videos})
    result['flight_separation'] = ('Video IDs are disjoint. Flight identities are not supplied by the manifest. '
                                  'Shared filename prefixes mean independence across flights is unproven.')
    result['duplicate_image_files'] = [v for v in hashes.values() if len(v)>1]
    result['duplicate_decoded_images'] = [v for v in pixel_hashes.values() if len(v)>1]
    result['cross_split_decoded_duplicates'] = [v for v in pixel_hashes.values()
        if any(p.startswith('train/') for p in v) and any(p.startswith('val/') for p in v)]
    result['dataset_digest'] = hashlib.sha256(json.dumps(sorted(hashes.items())).encode()).hexdigest()
    result['data_yaml'] = (root/'data.yaml').read_text()
    result['source'] = str(args.source) if args.source else None
    write(args.out/'audit.json', result)


def overfit(args):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    ds = BirdsaiDataset(args.data, 'train', args.size, 0, 42, augment=False)
    # Spread examples across source videos, retaining both classes.
    groups = defaultdict(list)
    for i, (p, _, boxes) in enumerate(ds.records):
        if any(b[0] == 1 for b in boxes):
            groups['human'].append(i)
        elif boxes:
            groups['animal'].append(i)
    chosen = []
    for name in ('human', 'animal'):
        pool = groups[name]
        chosen += [pool[round(k*(len(pool)-1)/7)] for k in range(8)]
    samples = [ds[i] for i in chosen]
    loader = DataLoader(samples, batch_size=4, shuffle=True, collate_fn=collate)
    eval_loader = DataLoader(samples, batch_size=4, collate_fn=collate)
    model = ThermalScoutReal().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    initial_loss = eval_epoch(model, eval_loader, args.device, objectness=args.objectness, human_weight=4)
    initial = metrics(predict(model, samples, args.device))
    history = []
    best = (-1, -1)
    for epoch in range(1, args.epochs+1):
        loss, collision = train_epoch(model, loader, optimizer, args.device, objectness=args.objectness, human_weight=4)
        if epoch % 20 == 0 or epoch == 1 or epoch == args.epochs:
            measured = metrics(predict(model, samples, args.device))
            eval_loss, _ = eval_epoch(model, eval_loader, args.device, objectness=args.objectness, human_weight=4)
            row = dict(epoch=epoch, train_loss=loss, eval_loss=eval_loss, collisions=collision, **measured)
            history.append(row)
            print(json.dumps(row), flush=True)
            quality = (measured['map50'], min(v['precision'] for v in measured['class_metrics'].values()))
            if quality > best:
                best = quality
                torch.save(dict(model=model.state_dict(), image_size=args.size, classes=CLASSES,
                                epoch=epoch, selection='memorization mAP50, then minimum class precision',
                                map50=best[0], objectness_loss=args.objectness),
                           args.out/'thermal_scout_real_weights.pt')
            if measured['map50'] >= .95 and all(v['recall'] >= .95 and v['precision'] >= .90
                                               for v in measured['class_metrics'].values()):
                break
    ck = torch.load(args.out/'thermal_scout_real_weights.pt', weights_only=True, map_location=args.device)
    model.load_state_dict(ck['model'])
    entries = predict(model, samples, args.device)
    final = metrics(entries)
    scores = [p[1] for e in entries for p in e['predictions'] if p[1] >= .25]
    passed = final['map50'] >= .95 and all(v['recall'] >= .95 and v['precision'] >= .90
                                         for v in final['class_metrics'].values()) and len(set(scores)) > 1
    write(args.out/'predictions.json', entries)
    write(args.out/'history.json', history)
    write(args.out/'summary.json', dict(passed=passed, scope='Training-set memorization only; no generalization claim',
          size=args.size, seed=42, device=args.device, images=len(samples), augmentation=False,
          objectness=args.objectness,
          checkpoint_epoch=ck['epoch'], initial_loss=initial_loss[0], initial=initial, final=final,
          score_min=min(scores, default=None), score_max=max(scores, default=None),
          unique_scores=len(set(scores)), selected_images=[Path(s[2]).name for s in samples],
          checkpoint_sha256=digest(args.out/'thermal_scout_real_weights.pt')))
    print('Memorization gate:', passed, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['audit', 'overfit'])
    p.add_argument('--data', type=Path, default=Path('data/processed/yolo_real'))
    p.add_argument('--source', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--objectness', choices=['baseline','hard-negative','focal'], default='baseline')
    p.add_argument('--device', default='cuda', choices=['cuda','cpu'])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out/'config.json', dict(mode=args.mode, data=str(args.data.resolve()),
        source=str(args.source.resolve()) if args.source else None, size=args.size,
        epochs=args.epochs, objectness=args.objectness, device=args.device,
        python=platform.python_version(), torch=str(torch.__version__), command=sys.argv,
        source_sha256={str(p):digest(p) for directory in ('thermalscout','scripts')
                       for p in Path(directory).glob('*.py')}))
    torch.set_num_threads(4)
    (audit if args.mode == 'audit' else overfit)(args)


if __name__ == '__main__':
    main()
