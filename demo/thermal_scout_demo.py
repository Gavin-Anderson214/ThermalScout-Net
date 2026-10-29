"""ThermalScout-Net: honest, standalone proof-of-concept demo.

This script trains a small CNN from scratch on GENERATED thermal-like images,
then identifies simulated humans and animals in a NEW synthetic test image.
It is NOT the final BIRDSAI-trained detector, and the thermal data are synthetic.

Requirements: torch, numpy, matplotlib
Run: python thermal_scout_demo.py
Fast: python thermal_scout_demo.py --quick
Train again: python thermal_scout_demo.py --retrain
Save without opening window: python thermal_scout_demo.py --no-show
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

SIZE = 80
CLASS_NAMES = {1: "Human", 2: "Animal"}
COLORS = {1: "#53c6ff", 2: "#a8ff78"}
OUTPUT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = OUTPUT_DIR / "thermal_scout_demo_weights.pt"
PREVIEW_PATH = OUTPUT_DIR / "thermal_scout_demo_result.png"


def ellipse(xx, yy, cx, cy, rx, ry):
    """Boolean mask for one axis-aligned ellipse."""
    return ((xx - cx) / max(rx, 0.3)) ** 2 + ((yy - cy) / max(ry, 0.3)) ** 2 <= 1.0


def generate_thermal_frame(rng: np.random.Generator, n_objects=None):
    """Synthetic warm silhouettes on noisy cooler backgrounds, plus true masks."""
    h = w = SIZE
    yy, xx = np.mgrid[:h, :w]
    # Background gradient, sensor-like grain, and harmless warm clutter.
    base = 0.14 + 0.0009 * xx + 0.0006 * yy
    img = base + rng.normal(0, 0.038, (h, w))
    img = img.astype(np.float32)
    for _ in range(int(rng.integers(0, 4))):
        cx, cy = rng.uniform(5, w-5), rng.uniform(5, h-5)
        radius = rng.uniform(3.5, 8.0)
        img += (rng.uniform(0.05, 0.12) * np.exp(-((xx-cx)**2 + (yy-cy)**2)/(2*radius**2))).astype(np.float32)

    labels = np.zeros((h, w), dtype=np.int64)
    objects = []
    n = int(n_objects) if n_objects is not None else int(rng.integers(2, 5))
    for i in range(n):
        category = 1 if i % 2 == 0 else 2
        if rng.random() < 0.25:
            category = int(rng.choice([1, 2]))
        for _attempt in range(45):
            cx, cy = int(rng.integers(15, w-15)), int(rng.integers(15, h-15))
            if all(np.hypot(cx - ox, cy - oy) > 23 for ox, oy in objects):
                break
        else:
            continue
        objects.append((cx, cy))
        if category == 1:
            # Upright heat signature: head, torso, and two legs.
            scale = rng.uniform(0.85, 1.23)
            m = ellipse(xx, yy, cx, cy - 1, 3.3*scale, 7.2*scale)
            m |= ellipse(xx, yy, cx, cy-10.0*scale, 2.45*scale, 2.5*scale)
            m |= ellipse(xx, yy, cx-1.9*scale, cy+7*scale, 1.2*scale, 5*scale)
            m |= ellipse(xx, yy, cx+1.9*scale, cy+7*scale, 1.2*scale, 5*scale)
        else:
            # Four-legged animal signature: horizontal body, head and legs.
            scale = rng.uniform(0.8, 1.15)
            m = ellipse(xx, yy, cx-2*scale, cy, 9*scale, 3.8*scale)
            m |= ellipse(xx, yy, cx+8*scale, cy-2*scale, 3.0*scale, 3*scale)
            m |= ellipse(xx, yy, cx-7*scale, cy+4*scale, 1.3*scale, 4*scale)
            m |= ellipse(xx, yy, cx+3*scale, cy+4*scale, 1.3*scale, 4*scale)
        # If any objects overlap, preserve the already labeled pixels.
        m &= labels == 0
        img[m] = rng.uniform(0.70, 0.97) + rng.normal(0, 0.025, int(m.sum()))
        labels[m] = category

    img = np.clip(img, 0, 1)
    return img.astype(np.float32), labels


class SyntheticThermalDataset(Dataset):
    """Create many simulated frames for introductory CNN training."""
    def __init__(self, count, seed):
        self.count = count
        self.seed = seed

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        # Deterministic dataset; changing the index yields a new scene.
        rng = np.random.default_rng(self.seed + i * 193)
        image, mask = generate_thermal_frame(rng)
        return torch.from_numpy(image[None]), torch.from_numpy(mask)


class ConvBlock(nn.Module):
    """A simple convolution block for extracting local image patterns."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class ThermalScoutNetDemo(nn.Module):
    """Small custom encoder-decoder CNN; class masks become detection boxes.

    NOTE: This teaching prototype uses segmentation + connected components,
    rather than the anchor-free detection heads proposed for final ThermalScout-Net.
    """
    def __init__(self):
        super().__init__()
        self.encoder1 = ConvBlock(1, 8)
        self.encoder2 = ConvBlock(8, 16)
        self.bottleneck = ConvBlock(16, 32)
        self.decoder2 = ConvBlock(32 + 16, 16)
        self.decoder1 = ConvBlock(16 + 8, 8)
        self.classifier = nn.Conv2d(8, 3, 1)  # background, human, animal

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        center = self.bottleneck(F.max_pool2d(e2, 2))
        up2 = F.interpolate(center, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.decoder2(torch.cat([up2, e2], dim=1))
        up1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.decoder1(torch.cat([up1, e1], dim=1))
        return self.classifier(d1)


def train_model(model, device, epochs, samples):
    """Teach the synthetic CNN over a number of training epochs."""
    loader = DataLoader(SyntheticThermalDataset(samples, seed=123), batch_size=20, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.35, 2.0, 2.0], device=device))
    history = []
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(inputs)
        avg = running / samples
        history.append(avg)
        print(f"Epoch {epoch+1:02d}/{epochs}: training loss = {avg:.4f}", flush=True)
    return history


def connected_components(binary_mask, min_area=14):
    """Find objects without additional packages such as OpenCV or SciPy."""
    h, w = binary_mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []
    for y in range(h):
        for x in range(w):
            if not binary_mask[y, x] or visited[y, x]:
                continue
            q = deque([(x, y)])
            visited[y, x] = True
            points = []
            while q:
                xx, yy = q.popleft()
                points.append((xx, yy))
                for nx, ny in ((xx-1, yy), (xx+1, yy), (xx, yy-1), (xx, yy+1)):
                    if 0 <= nx < w and 0 <= ny < h and binary_mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((nx, ny))
            if len(points) >= min_area:
                components.append(points)
    return components


def get_boxes(label_mask, probs=None):
    """Create bounding boxes around predicted object regions."""
    boxes = []
    for cls in (1, 2):
        for points in connected_components(label_mask == cls):
            coords = np.asarray(points)
            x0, y0 = coords.min(axis=0)
            x1, y1 = coords.max(axis=0)
            confidence = None
            if probs is not None:
                confidence = float(np.mean(probs[cls, coords[:, 1], coords[:, 0]]))
            boxes.append((int(x0), int(y0), int(x1), int(y1), cls, confidence))
    return boxes


def annotate(ax, image, boxes, title):
    """Draw object labels and boxes for a demonstration figure."""
    ax.imshow(image, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=12, weight="bold")
    for x0, y0, x1, y1, cls, confidence in boxes:
        color = COLORS[cls]
        ax.add_patch(Rectangle((x0-.5, y0-.5), x1-x0+1, y1-y0+1,
                               fill=False, edgecolor=color, linewidth=1.8))
        name = CLASS_NAMES[cls]
        if confidence is not None:
            name += f" {confidence:.0%}"
        ax.text(x0, max(2, y0-1), name, color="black", fontsize=7, va="bottom",
                bbox={"facecolor": color, "edgecolor": "none", "pad": 1.5})
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    """Train or load the synthetic demo, then show a sample prediction."""
    parser = argparse.ArgumentParser(description="Custom CNN thermal object detection teaching demo")
    parser.add_argument("--quick", action="store_true", help="Faster, less accurate training: 2 epochs")
    parser.add_argument("--retrain", action="store_true", help="Train again and overwrite saved weights")
    parser.add_argument("--no-show", action="store_true", help="Only save output PNG")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs (default: 8)")
    parser.add_argument("--seed", type=int, default=202699, help="Random seed for an unseen synthetic demo frame")
    args = parser.parse_args()

    torch.manual_seed(7)
    np.random.seed(7)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}. This demo learns from SYNTHETIC images, not BIRDSAI.", flush=True)
    model = ThermalScoutNetDemo().to(device)
    print(f"Custom trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    if WEIGHTS_PATH.exists() and not args.retrain and not args.quick:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
        print("Loaded saved demo network weights.")
    else:
        epochs, samples = (2, 240) if args.quick else (args.epochs, 600)
        history = train_model(model, device, epochs, samples)
        if not args.quick:
            torch.save(model.cpu().state_dict(), WEIGHTS_PATH)
            model.to(device)
            print(f"Saved demo network weights: {WEIGHTS_PATH}")

    # Unseen synthetic sample has a different random seed than training images.
    sample, truth = generate_thermal_frame(np.random.default_rng(args.seed), n_objects=3)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(sample)[None, None].to(device))
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        label_mask = np.argmax(probs, axis=0)
        label_mask[np.max(probs[1:], axis=0) < 0.60] = 0
    truth_boxes = get_boxes(truth)
    predicted_boxes = get_boxes(label_mask, probs)

    fig, ax = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    annotate(ax[0], sample, truth_boxes, "Ground truth (synthetic labels)")
    annotate(ax[1], sample, predicted_boxes, "Custom CNN predictions")
    ax[2].imshow(label_mask, vmin=0, vmax=2, cmap=plt.matplotlib.colors.ListedColormap(["#161b28", COLORS[1], COLORS[2]]), interpolation="nearest")
    ax[2].set_title("Predicted class map", fontsize=12, weight="bold")
    ax[2].set_xticks([]); ax[2].set_yticks([])
    fig.suptitle("ThermalScout-Net proof of concept | synthetic data, not real drone footage", fontsize=13, weight="bold")
    fig.savefig(PREVIEW_PATH, dpi=160, bbox_inches="tight")
    print(f"Predicted {len(predicted_boxes)} object(s), true count {len(truth_boxes)}.")
    print(f"Saved image: {PREVIEW_PATH}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
