# ThermalScout-Net | COSC 2345 Introduction to AI
# Each module has one job. The network learns from annotated thermal images.
# IMPORTANT: TestReal is not used during training or development evaluation.

"""Run a quick software check or train on the full prepared development split.

Use project_manager.py at the repository root for automatic experiment logging.
"""

import argparse
import itertools
import signal
import csv
import json
import math
import platform
import hashlib
from io import BytesIO
import sys
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import ThermalScoutReal, CLASSES
from .data import BirdsaiDataset, TiledBirdsaiDataset, collate
from .training import train_epoch, eval_epoch
from .inference import create_preview


def load_pretrained_features(model, path, size, device):
    """Load the feature layers while retaining the newly initialized detection head."""
    content = path.read_bytes()
    checkpoint = torch.load(BytesIO(content), map_location=device, weights_only=True)
    state = model.state_dict()
    expected = {name for name in state if not name.startswith("head.")}
    features = checkpoint.get("features")
    if checkpoint.get("image_size") != size or not isinstance(features, dict) or features.keys() != expected:
        raise ValueError("Feature checkpoint must match the image size and all detector feature layers.")
    model.load_state_dict({**state, **features})
    return hashlib.sha256(content).hexdigest()


# --- Parse settings, select development data, train from random weights and save the lowest-validation-loss checkpoint.
def main():
    """Read training options, train the model, save the best weights and loss history."""
    p = argparse.ArgumentParser(description='Train a custom ThermalScout-Net detector')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--quick', action='store_true', help='Small real-data training smoke test (default)')
    mode.add_argument('--full', action='store_true', help='Use the entire prepared development sample')
    p.add_argument('--data', type=Path, default=Path('data/processed/yolo_real'))
    p.add_argument('--out', type=Path, default=Path('training_runs/standalone'))
    p.add_argument('--size', type=int, default=256, help='Input image side in pixels (multiple of 8)')
    p.add_argument('--tile', action='store_true', help='Train on overlapping native-resolution image tiles')
    p.add_argument('--verified-backgrounds', type=Path, help='Reviewed empty images from training videos only')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--threshold', type=float, default=0.25)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--objectness', choices=['baseline', 'focal', 'hard-negative'], default='baseline', help='Negative objectness loss for controlled experiment')
    p.add_argument('--human-weight', type=float, default=2.0, help='Human class cross-entropy weight; animal stays 1')
    p.add_argument('--background-weight', type=float, default=0.25, help='Weight of the negative objectness loss')
    p.add_argument('--pretrained-features', type=Path, help='Optional weights from thermal patch pretraining')
    p.add_argument('--continuous', action='store_true', help='Train until Ctrl+C; save after finishing the current epoch')
    p.add_argument('--patience', type=int, default=None, help='Stop after this many epochs without lower validation loss')
    args = p.parse_args()
    if args.continuous and args.epochs is not None:
        p.error('Use --continuous or --epochs, not both')
    if args.patience is not None and args.patience < 1:
        p.error('--patience must be positive')
    if not math.isfinite(args.human_weight) or args.human_weight <= 0:
        p.error('--human-weight must be finite and positive')
    if not math.isfinite(args.background_weight) or args.background_weight <= 0:
        p.error('--background-weight must be finite and positive')
    if args.size < 64 or args.size % 8 != 0 or args.batch < 1:
        p.error('--size must be >=64 and divisible by 8; --batch >=1')
    root = args.data.expanduser().resolve()
    for split in ('train', 'val'):
        if not (root / 'images' / split).is_dir() or not (root / 'labels' / split).is_dir():
            p.error(f"Could not find {split} images/labels under {root}. Run birdsai_prepare.py --mode prepare first.")
    # Seed sampling and initialization for repeatability; GPU kernels may still vary.
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    if device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable in this PyTorch installation; use --device cpu or install CUDA PyTorch.')
    full = args.full
    # Quick mode checks the pipeline on a small sample; full uses the prepared split.
    tr_limit, va_limit = (0, 0) if full else (160, 40)
    epochs = args.epochs if args.epochs is not None else (10 if full else 2)
    dataset_class = TiledBirdsaiDataset if args.tile else BirdsaiDataset
    train = dataset_class(root, 'train', args.size, tr_limit, args.seed, augment=True,
                          background_dir=args.verified_backgrounds)
    val = dataset_class(root, 'val', args.size, va_limit, args.seed + 1, augment=False)
    tr_loader = DataLoader(train, batch_size=args.batch, shuffle=True,
                           num_workers=0, collate_fn=collate)
    va_loader = DataLoader(val, batch_size=args.batch, shuffle=False,
                           num_workers=0, collate_fn=collate)
    model = ThermalScoutReal().to(device)
    initialization = {}
    if args.pretrained_features:
        path = args.pretrained_features.expanduser().resolve()
        try:
            digest = load_pretrained_features(model, path, args.size, device)
        except (OSError, ValueError, RuntimeError) as error:
            p.error(f'Cannot load pretrained features: {error}')
        initialization = {'initialization': 'thermal patch pretraining', 'pretrained_features_sha256': digest}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'thermal_scout_real_weights.pt').exists():
        p.error('Output already contains a checkpoint; choose a fresh --out directory')
    config = {k: v for k, v in vars(args).items() if k != 'pretrained_features'}
    config.update({'data': str(root), 'out': str(out), 'device': device,
              'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__,
              'command': sys.argv, 'optimizer': 'AdamW', 'weight_decay': 1e-4,
              'train_images': len(train), 'val_images': len(val), 'epochs': epochs,
              'verified_backgrounds': str(args.verified_backgrounds.resolve()) if args.verified_backgrounds else None,
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('thermalscout').glob('*.py')}})
    if initialization:
        config.update(initialization, pretrained_features=str(path))
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'Device: {device} | Trainable CUSTOM parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Mode: {"FULL" if full else "QUICK"} | images train={len(train)} val={len(val)} | image size={args.size}')
    print(f'Annotations in selected images: train animal={train.label_counts[0]}, human={train.label_counts[1]}; '
          f'val animal={val.label_counts[0]}, human={val.label_counts[1]}')
    print(f'Unlabeled selected images: train={train.num_empty}; val={val.num_empty}')
    if initialization:
        print('Fine-tuning thermal features with labeled human and animal boxes.')
    else:
        print('This is training from RANDOM INITIALIZATION on the supplied thermal frames. No pretrained model.')
    print('Validation uses the supplied development split; this is not an official test score.')
    print(f'Objectness loss: {args.objectness}; human class weight: {args.human_weight}', flush=True)
    best = float('inf')
    best_epoch = 0
    history = []
    stopping = False
    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
        print('Stopping after the current epoch. Please wait.', flush=True)
    if args.continuous:
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        print('Continuous training enabled. Press Ctrl+C to stop.', flush=True)

    def write_history():
        # Keep the dashboard and interrupted runs useful while continuous training is active.
        tmp = out / 'loss_history.csv.tmp'
        with tmp.open('w', newline='', encoding='utf-8') as fp:
            writer = csv.writer(fp)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_box_collisions', 'val_box_collisions'])
            writer.writerows(history)
        tmp.replace(out / 'loss_history.csv')

    epoch_numbers = itertools.count(1) if args.continuous else range(1, epochs + 1)
    for ep in epoch_numbers:
        tl, tc = train_epoch(model, tr_loader, optimizer, device, args.objectness, args.human_weight, args.background_weight)
        vl, vc = eval_epoch(model, va_loader, device, args.objectness, args.human_weight, args.background_weight)
        print(f'Epoch {ep:02d}/{'continuous' if args.continuous else epochs}: train loss={tl:.4f}, val loss={vl:.4f} '
              f'| same-cell box collisions train={tc}, val={vc}', flush=True)
        history.append((ep, tl, vl, tc, vc))
        write_history()
        # Select weights by validation loss, not by measured detection mAP.
        if vl < best:
            best = vl
            best_epoch = ep
            torch.save({'model': model.state_dict(), 'image_size': args.size,
                        'classes': CLASSES, 'epoch': ep, 'val_loss': vl, 'tile': args.tile,
                        'objectness_loss': args.objectness,
                        'human_weight': args.human_weight, 'background_weight': args.background_weight, **initialization}, out / 'thermal_scout_real_weights.pt')
        if stopping:
            break
        if args.patience is not None and ep - best_epoch >= args.patience:
            print(f'Early stopping: no lower validation loss for {args.patience} epochs.', flush=True)
            break
    write_history()
    # Reload the best epoch rather than using the final epoch for the preview.
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
