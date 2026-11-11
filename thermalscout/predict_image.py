"""Run the custom CNN on one image and return boxes in original-image pixels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .inference import infer
from .model import CLASSES, COLORS, ThermalScoutReal


def prepare_image(image, size):
    """Match BirdsaiDataset's grayscale, black padding and normalization."""
    if size < 8 or size % 8:
        raise ValueError('Image size must be a positive multiple of 8')
    width, height = image.size
    scale = min(size / width, size / height)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    offset = ((size - resized[0]) // 2, (size - resized[1]) // 2)
    canvas = Image.new('L', (size, size), 0)
    canvas.paste(image.convert('L').resize(resized, Image.Resampling.BILINEAR), offset)
    pixels = np.asarray(canvas, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(pixels[None]), dict(original=[width, height],
                                               resized=list(resized), offset=list(offset), size=size)


def original_boxes(detections, geometry):
    """Undo letterboxing, clip to the image, and discard padding-only boxes."""
    width, height = geometry['original']
    rw, rh = geometry['resized']
    ox, oy = geometry['offset']
    size = geometry['size']
    result = []
    for class_id, score, (x1, y1, x2, y2) in detections:
        box = [min(width, max(0, (x1 * size - ox) * width / rw)),
               min(height, max(0, (y1 * size - oy) * height / rh)),
               min(width, max(0, (x2 * size - ox) * width / rw)),
               min(height, max(0, (y2 * size - oy) * height / rh))]
        if box[2] > box[0] and box[3] > box[1]:
            result.append(dict(class_id=class_id, label=CLASSES[class_id],
                               confidence=score, box_xyxy_pixels=box))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New output directory; never overwritten')
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('Threshold must be between 0 and 1')
    if args.out.exists():
        parser.error('Output directory already exists; choose a new folder')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = ThermalScoutReal().to(device)
    model.load_state_dict(checkpoint['model'])
    with Image.open(args.image) as source:
        image = source.convert('RGB')
    tensor, geometry = prepare_image(image, int(checkpoint.get('image_size', 256)))
    detections = original_boxes(infer(model, tensor, device, args.threshold), geometry)
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for detection in detections:
        box = detection['box_xyxy_pixels']
        color = COLORS[detection['class_id']]
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0], max(0, box[1] - 12)),
                  f"{detection['label']} {detection['confidence']:.2f}", fill=color)
    record = dict(image=args.image.name, checkpoint=args.checkpoint.name,
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  image_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
                  device=device, threshold=args.threshold, geometry=geometry,
                  max_candidates=500, max_detections=40, local_peaks=False,
                  detections=detections,
                  note='Model predictions, not verified labels. Still-image demonstration; not live drone detection.')
    args.out.mkdir(parents=True, exist_ok=False)
    preview.save(args.out / 'preview.png')
    (args.out / 'detections.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'Saved {len(detections)} predictions on {device} to {args.out}')


if __name__ == '__main__':
    main()
