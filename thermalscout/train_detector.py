# ThermalScout-Net | COSC 2345 Introduction to AI
# Each module has one job. The network learns from annotated thermal images.
# IMPORTANT: TestReal is not used during training or development evaluation.

"""Run a quick software check or train on the full prepared development split.

Use project_manager.py at the repository root for automatic experiment logging.
"""

import argparse
import csv
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import ThermalScoutReal, CLASSES
from .data import BirdsaiDataset, collate
from .training import train_epoch, eval_epoch
from .inference import create_preview

def main():
    """Read training options, train the model, save the best weights and loss history."""
    p = argparse.ArgumentParser(description='Train a custom ThermalScout-Net detector from scratch')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--quick', action='store_true', help='Small real-data training smoke test (default)')
    mode.add_argument('--full', action='store_true', help='Use the entire prepared development sample')
    p.add_argument('--data', type=Path, default=Path('data/processed/yolo_real'))
    p.add_argument('--out', type=Path, default=Path('training_runs/standalone'))
    p.add_argument('--size', type=int, default=256, help='Input image side in pixels (multiple of 8)')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--threshold', type=float, default=0.25)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.size < 64 or args.size % 8 != 0 or args.batch < 1:
        p.error('--size must be >=64 and divisible by 8; --batch >=1')
    root = args.data.expanduser().resolve()
    for split in ('train', 'val'):
        if not (root / 'images' / split).is_dir() or not (root / 'labels' / split).is_dir():
            p.error(f"Could not find {split} images/labels under {root}. Run birdsai_prepare.py --mode prepare first.")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    if device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable in this PyTorch installation; use --device cpu or install CUDA PyTorch.')
    full = args.full
    tr_limit, va_limit = (0, 0) if full else (160, 40)
    epochs = args.epochs if args.epochs is not None else (10 if full else 2)
    train = BirdsaiDataset(root, 'train', args.size, tr_limit, args.seed, augment=True)
    val = BirdsaiDataset(root, 'val', args.size, va_limit, args.seed + 1, augment=False)
    tr_loader = DataLoader(train, batch_size=args.batch, shuffle=True,
                           num_workers=0, collate_fn=collate)
    va_loader = DataLoader(val, batch_size=args.batch, shuffle=False,
                           num_workers=0, collate_fn=collate)
    model = ThermalScoutReal().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f'Device: {device} | Trainable CUSTOM parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Mode: {"FULL" if full else "QUICK"} | images train={len(train)} val={len(val)} | image size={args.size}')
    print(f'Annotations in selected images: train animal={train.label_counts[0]}, human={train.label_counts[1]}; '
          f'val animal={val.label_counts[0]}, human={val.label_counts[1]}')
    print(f'Unlabeled selected images: train={train.num_empty}; val={val.num_empty}')
    print('This is training from RANDOM INITIALIZATION on real BIRDSAI frames. No pretrained model.')
    print('Validation is for development; this DOES NOT use the official TestReal partition.')
    best = float('inf')
    history = []
    for ep in range(1, epochs + 1):
        tl, tc = train_epoch(model, tr_loader, optimizer, device)
        vl, vc = eval_epoch(model, va_loader, device)
        print(f'Epoch {ep:02d}/{epochs}: train loss={tl:.4f}, val loss={vl:.4f} '
              f'| same-cell box collisions train={tc}, val={vc}', flush=True)
        history.append((ep, tl, vl, tc, vc))
        if vl < best:
            best = vl
            torch.save({'model': model.state_dict(), 'image_size': args.size,
                        'classes': CLASSES, 'epoch': ep, 'val_loss': vl}, out / 'thermal_scout_real_weights.pt')
    with (out / 'loss_history.csv').open('w', newline='', encoding='utf-8') as fp:
        writer = csv.writer(fp)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_box_collisions', 'val_box_collisions'])
        writer.writerows(history)
    saved = torch.load(out / 'thermal_scout_real_weights.pt', map_location=device, weights_only=True)
    model.load_state_dict(saved['model'])
    n, name = create_preview(model, val, device, out / 'real_detection_preview.png', args.threshold)
    print(f'Best checkpoint epoch {saved["epoch"]}, val loss {best:.4f}')
    print(f'Preview: {out / "real_detection_preview.png"} ({n} detections on {name})')
    print(f'Weights: {out / "thermal_scout_real_weights.pt"}')
    print(f'History: {out / "loss_history.csv"}')
    print('CAUTION: This is a first custom detector baseline, not a validated mAP result.')

if __name__ == "__main__":
    main()
