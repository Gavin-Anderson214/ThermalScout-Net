"""Rank labeled training frames for human review using missed and false boxes.

This produces a review queue, not new labels or a score on unseen footage.
Only prepared BIRDSAI training frames are read; TestReal is never opened.
"""
import argparse
import csv
import json
from pathlib import Path

import torch

from thermalscout.data import BirdsaiDataset
from thermalscout.evaluate_detector import get_counts
from thermalscout.inference import infer
from thermalscout.model import ThermalScoutReal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/processed/yolo_real'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--limit', type=int, default=0, help='0 reviews all prepared training frames')
    parser.add_argument('--per-video', type=int, default=10)
    parser.add_argument('--min-gap', type=int, default=50, help='Source-frame spacing within each video')
    args = parser.parse_args()
    if args.limit < 0 or args.per_video < 1 or args.min_gap < 0:
        parser.error('Invalid limit, per-video count or frame gap')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if checkpoint.get('tile'):
        parser.error('Use a full-frame checkpoint for this review queue')
    model = ThermalScoutReal().to(args.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    root = args.data.resolve()
    dataset = BirdsaiDataset(root, 'train', int(checkpoint['image_size']), args.limit, 42)
    split = json.loads((root / 'split_videos.json').read_text(encoding='utf-8'))
    allowed = set(split['train_videos'])
    selected = []
    for i, (path, _, _) in enumerate(dataset.records):
        video, frame = path.stem.rsplit('__f', 1)
        if video not in allowed:
            raise ValueError(f'Frame outside the training videos: {path}')
        image, boxes, _ = dataset[i]
        entry = {'ground_truth': boxes, 'predictions': infer(model, image, args.device, threshold=0.25)}
        animal, human = (get_counts([entry], c) for c in (0, 1))
        missed_human, missed_animal = human['gt']-human['tp'], animal['gt']-animal['tp']
        score = 3*missed_human + missed_animal + 0.1*(animal['fp']+human['fp'])
        selected.append(dict(score=score, video=video, frame=int(frame), image=str(path),
                             missed_human=missed_human, missed_animal=missed_animal,
                             false_human=human['fp'], false_animal=animal['fp']))
        if (i+1) % 100 == 0:
            print(f'Scored {i+1}/{len(dataset)} training frames', flush=True)
    selected.sort(key=lambda x: -x['score'])
    kept = []
    by_video = {}
    for row in selected:
        frames = by_video.setdefault(row['video'], [])
        if len(frames) >= args.per_video or any(abs(row['frame']-f) < args.min_gap for f in frames):
            continue
        frames.append(row['frame'])
        kept.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=('score','video','frame','image','missed_human',
                                                  'missed_animal','false_human','false_animal'))
        writer.writeheader()
        writer.writerows(kept)
    print(f'Saved {len(kept)} diverse training-frame review candidates to {args.out}')
    print('Inspect each frame and its labels manually before adding any negative or corrected example.')


if __name__ == '__main__':
    main()
