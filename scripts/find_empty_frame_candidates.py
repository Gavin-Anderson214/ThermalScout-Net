"""List possible empty BIRDSAI training frames for manual full-resolution review.

Absent source annotation rows are only a candidate signal, never proof of an
empty scene. This tool does not copy images or create training labels.
"""
import argparse
import csv
import json
import random
from pathlib import Path

from scripts.birdsai_prepare import collect_sequences, find_train_real, numeric_tail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='BIRDSAI TrainReal directory or its parent')
    parser.add_argument('--data', type=Path, default=Path('data/processed/yolo_real'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--per-video', type=int, default=5)
    parser.add_argument('--min-gap', type=int, default=100)
    args = parser.parse_args()
    if args.per_video < 1 or args.min_gap < 0:
        parser.error('Choose positive per-video count and nonnegative frame gap')
    train_real = find_train_real(args.source)
    if train_real is None or not (train_real/'annotations').is_dir():
        parser.error('Could not find TrainReal images and annotations')
    split = json.loads((args.data.resolve()/'split_videos.json').read_text(encoding='utf-8'))
    train_videos = set(split['train_videos'])
    validation_videos = set(split['validation_videos'])
    if train_videos & validation_videos:
        parser.error('Train/validation videos overlap')
    rng = random.Random(42)
    rows = []
    for sequence in collect_sequences(train_real, None):
        if sequence.get('error') or sequence['name'] not in train_videos:
            continue
        offset = sequence['offset']
        labeled = set(sequence['annotations'])
        candidates = [p for p in sequence['images']
                      if numeric_tail(p) is not None and numeric_tail(p)-offset not in labeled]
        rng.shuffle(candidates)
        chosen = []
        for path in candidates:
            frame = numeric_tail(path)-offset
            if frame < 0 or any(abs(frame-other) < args.min_gap for other in chosen):
                continue
            chosen.append(frame)
            rows.append({'video':sequence['name'],'frame':frame,'image':str(path),
                         'reviewed_empty':'NO'})
            if len(chosen) >= args.per_video:
                break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=('video','frame','image','reviewed_empty'))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Found {len(rows)} candidate frames from {len(train_videos)} allowed training videos: {args.out}')
    print('Open each original image at full resolution and confirm every human and animal is absent before using it as an empty training example.')


if __name__ == '__main__':
    main()
