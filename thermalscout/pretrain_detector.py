"""Teach the detector's feature layers to reconstruct hidden thermal patches."""

import argparse
import json
from pathlib import Path
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import BirdsaiDataset, collate
from .model import ThermalScoutReal


PATCH_SIZE = 16
MASK_PROBABILITY = 0.25


def reconstruction_loss(model, decoder, images):
    batch, _, height, width = images.shape
    patches = torch.rand(batch, 1, height // PATCH_SIZE, width // PATCH_SIZE, device=images.device)
    mask = (patches < MASK_PROBABILITY).repeat_interleave(PATCH_SIZE, dim=2)
    mask = mask.repeat_interleave(PATCH_SIZE, dim=3)
    # The loader letterboxes the image with zeros. Skip blank rows and columns.
    has_signal = images.ne(0)
    inside = has_signal.any(dim=-1, keepdim=True) & has_signal.any(dim=-2, keepdim=True)
    hidden_pixels = mask & inside
    masked = images.masked_fill(mask, -1)
    prediction = decoder(model.features(masked))
    prediction = F.interpolate(prediction, size=(height, width), mode="bilinear", align_corners=False)
    error = (prediction.sigmoid() - images).abs()
    return error.masked_select(hidden_pixels).sum() / hidden_pixels.sum().clamp_min(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/yolo_real"))
    parser.add_argument("--out", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0 uses all training frames")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.size < 64 or args.size % PATCH_SIZE:
        parser.error(f"--size must be at least 64 and divisible by {PATCH_SIZE}")
    if args.epochs < 1 or args.batch < 1 or args.limit < 0:
        parser.error("Use positive epochs and batch size and a nonnegative frame limit")
    if args.out.exists():
        parser.error("Choose a fresh --out directory")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch installation")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = BirdsaiDataset(args.data, "train", args.size, args.limit, args.seed, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=collate)
    model = ThermalScoutReal().to(args.device)
    decoder = nn.Conv2d(model.head.in_channels, 1, kernel_size=1).to(args.device)
    model.head.requires_grad_(False)
    optimizer = torch.optim.AdamW([*model.parameters(), *decoder.parameters()], lr=0.001, weight_decay=1e-4)
    print(f"Pretraining on {len(dataset)} training frames using hidden patches on {args.device}.", flush=True)
    losses = []
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for images, _, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_loss(model, decoder, images.to(args.device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite reconstruction loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * images.size(0)
        losses.append(total / len(dataset))
        print(f"Pretraining epoch {epoch}/{args.epochs}: masked pixel error {losses[-1]:.4f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=False)
    features = {key: tensor.cpu() for key, tensor in model.state_dict().items() if not key.startswith("head.")}
    torch.save({"features": features, "image_size": args.size}, args.out / "thermal_features.pt")
    summary = {
        **vars(args),
        "data": str(args.data.resolve()),
        "out": str(args.out.resolve()),
        "training_images": len(dataset),
        "torch": str(torch.__version__),
        "patch_size": PATCH_SIZE,
        "mask_probability": MASK_PROBABILITY,
        "masked_pixel_error": losses,
    }
    (args.out / "pretrain.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved visual features: {args.out / 'thermal_features.pt'}")


if __name__ == "__main__":
    main()
