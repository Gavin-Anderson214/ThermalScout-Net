# ThermalScout-Net | COSC 2345 Introduction to AI
# Each module has one job. The network learns from annotated thermal images.
# IMPORTANT: TestReal is not used during training or development evaluation.

"""Read prepared BIRDSAI images and the boxes marking animals/people.

Input format: data.yaml + images/train, images/val, labels/train, labels/val.
We DO NOT read the original test footage here.
"""

import math
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# Image formats that the dataset loader understands.
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# --- Validate YOLO rows before training: class 0 is animal, class 1 is human; coordinates are normalized.
def read_boxes(txt: Path):
    """Read an image label file: each row has class, center x/y, width and height."""
    if not txt.exists():
        raise FileNotFoundError(f"Missing YOLO label file: {txt}")
    result = []
    for line_no, line in enumerate(txt.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid label {txt}:{line_no}: expected 'class x y w h'")
        c = int(parts[0])
        x, y, w, h = map(float, parts[1:])
        if c not in (0, 1) or not all(math.isfinite(t) for t in (x, y, w, h)):
            raise ValueError(f"Invalid numeric label in {txt}:{line_no}")
        if not (0 < x < 1 and 0 < y < 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Invalid normalized YOLO box in {txt}:{line_no}")
        result.append((c, x, y, w, h))
    return result

# --- Keep quick experiments repeatable while reserving some sample slots for frames containing humans.
def choose_subset(records, limit: int, seed: int):
    """Sample real images, reserving room for frames with the rarer human class."""
    if limit <= 0 or limit >= len(records):
        return records
    rng = random.Random(seed)
    human = [r for r in records if any(int(b[0]) == 1 for b in r[2])]
    count_human = min(len(human), max(1, limit // 3))
    keep = rng.sample(human, count_human)
    rest = [r for r in records if r not in keep]
    keep += rng.sample(rest, min(limit - len(keep), len(rest)))
    rng.shuffle(keep)
    return keep

# --- Connect image files to labels, then letterbox images and boxes into the same square coordinate system.
class BirdsaiDataset(Dataset):
    """Load images and labels, resize them, and optionally flip training images."""
    def __init__(self, root: Path, split: str, size: int, limit: int, seed: int,
                 augment: bool = False, background_dir: Path | None = None):
        """Find matching image and label files and select a repeatable sample."""
        self.size = size
        self.augment = augment
        imgs = sorted(p for p in (root / "images" / split).iterdir()
                      if p.is_file() and p.suffix.lower() in EXTS)
        if not imgs:
            raise FileNotFoundError(f"No images found in {root / 'images' / split}")
        records = [(p, root / "labels" / split / (p.stem + ".txt"),
                    read_boxes(root / "labels" / split / (p.stem + ".txt"))) for p in imgs]
        self.records = choose_subset(records, limit, seed)
        if background_dir is not None:
            if split != 'train':
                raise ValueError('Verified backgrounds may only be added to training')
            background_dir = background_dir.expanduser().resolve()
            if not background_dir.is_dir() or 'testreal' in str(background_dir).casefold():
                raise ValueError('Supply a reviewed training-background directory, never TestReal')
            backgrounds = sorted(p for p in background_dir.iterdir()
                                 if p.is_file() and p.suffix.lower() in EXTS)
            if not backgrounds:
                raise ValueError(f'No verified background images found in {background_dir}')
            self.records += [(p, None, []) for p in backgrounds]
        self.label_counts = [sum(sum(b[0] == c for b in r[2]) for r in self.records)
                             for c in (0, 1)]
        self.num_empty = sum(not r[2] for r in self.records)

    def __len__(self):
        """Tell PyTorch how many images are in this dataset."""
        return len(self.records)

    def __getitem__(self, idx):
        """Return one normalized thermal image plus its adjusted bounding boxes."""
        img_path, _, boxes = self.records[idx]
        with Image.open(img_path) as im:
            im = im.convert("L")
            orig_w, orig_h = im.size
            scale = min(self.size / orig_w, self.size / orig_h)
            new_w, new_h = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
            im = im.resize((new_w, new_h), Image.Resampling.BILINEAR)
            off_x = (self.size - new_w) // 2
            off_y = (self.size - new_h) // 2
            canvas = Image.new("L", (self.size, self.size), color=0)
            canvas.paste(im, (off_x, off_y))
            pixels = np.asarray(canvas, dtype=np.float32).copy() / 255.0

        # Padding changes object centers; resizing changes widths and heights.
        # Apply the same transformation to labels as to image pixels.
        # Letterbox-adjusted boxes: [c, xc, yc, w, h] all normalized to canvas.
        adjusted = []
        for c, x, y, w, h in boxes:
            ax = (x * orig_w * new_w / orig_w + off_x) / self.size
            ay = (y * orig_h * new_h / orig_h + off_y) / self.size
            aw = w * new_w / self.size
            ah = h * new_h / self.size
            adjusted.append((c, ax, ay, aw, ah))
        if self.augment and random.random() < 0.5:
            pixels = np.ascontiguousarray(pixels[:, ::-1])
            adjusted = [(c, 1 - x, y, w, h) for c, x, y, w, h in adjusted]
        return torch.from_numpy(pixels[None]), adjusted, str(img_path)

# --- Stack equal-sized images but preserve variable-length object lists for each frame.
def collate(batch):
    """Combine images into a batch while preserving each image’s list of boxes."""
    imgs, boxes, paths = zip(*batch)
    return torch.stack(imgs), boxes, paths


def tile_positions(length: int, side: int, overlap: float = 0.25):
    """Cover an image axis with overlapping native-resolution windows."""
    if side < 1 or length < side or not 0 <= overlap < 1:
        raise ValueError("Tile side must fit the image and overlap must be in [0, 1)")
    step = max(1, round(side * (1 - overlap)))
    return sorted(set([*range(0, length - side + 1, step), length - side]))


def tile_boxes(boxes, width, height, left, top, side):
    """Clip source-normalized boxes to a square tile, rejecting severe fragments."""
    result = []
    for c, x, y, bw, bh in boxes:
        x1, x2 = (x - bw / 2) * width, (x + bw / 2) * width
        y1, y2 = (y - bh / 2) * height, (y + bh / 2) * height
        a, b = max(x1, left), min(x2, left + side)
        d, e = max(y1, top), min(y2, top + side)
        if b <= a or e <= d or (b-a)*(e-d) < 0.5*bw*width*bh*height:
            continue
        result.append((c, ((a+b)/2-left)/side, ((d+e)/2-top)/side,
                       (b-a)/side, (e-d)/side))
    return result


class TiledBirdsaiDataset(BirdsaiDataset):
    """Train at native pixel scale, including empty crops from labeled frames."""
    def __init__(self, root, split, side, limit, seed, augment=False, background_dir=None):
        super().__init__(root, split, side, limit, seed, augment, background_dir)
        self.tiles = []
        self.label_counts = [0, 0]
        self.num_empty = 0
        for path, _, boxes in self.records:
            with Image.open(path) as im:
                width, height = im.size
            if min(width, height) < side:
                raise ValueError(f"Tile side {side} exceeds source image dimensions: {path}")
            for top in tile_positions(height, side):
                for left in tile_positions(width, side):
                    labels = tile_boxes(boxes, width, height, left, top, side)
                    self.tiles.append((path, left, top, labels))
                    self.num_empty += not labels
                    for label in labels:
                        self.label_counts[int(label[0])] += 1

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        path, left, top, labels = self.tiles[idx]
        with Image.open(path) as im:
            crop = im.convert('L').crop((left, top, left+self.size, top+self.size))
            pixels = np.asarray(crop, dtype=np.float32).copy() / 255.0
        if self.augment and random.random() < 0.5:
            pixels = np.ascontiguousarray(pixels[:, ::-1])
            labels = [(c, 1-x, y, w, h) for c, x, y, w, h in labels]
        return torch.from_numpy(pixels[None]), labels, f"{path}#tile={left},{top}"

