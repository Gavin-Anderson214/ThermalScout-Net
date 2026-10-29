# ThermalScout-Net | COSC 2345 Introduction to AI
# Each module has one job. The network learns from annotated thermal images.
# IMPORTANT: TestReal is not used during training or development evaluation.

"""Convert network outputs into boxes and create preview pictures."""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from .model import CLASSES, COLORS

def box_iou(box, other):
    """Compute the overlap of a proposed box with other boxes."""
    x1 = torch.maximum(box[0], other[:, 0]); y1 = torch.maximum(box[1], other[:, 1])
    x2 = torch.minimum(box[2], other[:, 2]); y2 = torch.minimum(box[3], other[:, 3])
    area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    oa = (other[:, 2] - other[:, 0]).clamp(min=0) * (other[:, 3] - other[:, 1]).clamp(min=0)
    return area / (a + oa - area + 1e-7)

def nms(boxes, scores, threshold=0.5):
    """Remove duplicate overlapping detections using non-maximum suppression."""
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = int(order[0]); keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        order = rest[box_iou(boxes[i], boxes[rest]) <= threshold]
    return keep

@torch.no_grad()
def infer(model, image, device, threshold=0.25):
    """Turn raw CNN maps into scored human/animal bounding boxes."""
    model.eval()
    pred = model(image.unsqueeze(0).to(device))[0].cpu()
    _, h, w = pred.shape
    prob = pred[0].sigmoid()
    clp = pred[5:7].softmax(dim=0)
    cls_prob, cls_id = clp.max(dim=0)
    score = prob * cls_prob
    ys, xs = torch.where(score >= threshold)
    if ys.numel() == 0:
        return []
    # Keep the most confident candidates across the whole image, not just
    # the first 500 found when scanning from top to bottom. This avoids a
    # positional bias when the network predicts many low-confidence boxes.
    if ys.numel() > 500:
        selected = torch.topk(score[ys, xs], k=500).indices
        ys, xs = ys[selected], xs[selected]
    ox = pred[1, ys, xs].sigmoid()
    oy = pred[2, ys, xs].sigmoid()
    bw = pred[3, ys, xs].clamp(-6, 5).exp()
    bh = pred[4, ys, xs].clamp(-6, 5).exp()
    x = (xs.float() + ox) / w; y = (ys.float() + oy) / h
    bw = bw / w; bh = bh / h
    box = torch.stack((x - bw/2, y - bh/2, x + bw/2, y + bh/2), dim=1).clamp(0, 1)
    sc, cl = score[ys, xs], cls_id[ys, xs]
    detections = []
    for c in range(2):
        indices = torch.where(cl == c)[0]
        if indices.numel() == 0:
            continue
        for k in nms(box[indices], sc[indices]):
            j = indices[k]
            detections.append((int(c), float(sc[j]), box[j].tolist()))
    return sorted(detections, key=lambda r: -r[1])[:40]

def create_preview(model, ds, device, output: Path, threshold: float):
    # Show what it predicts on one held-out validation frame.
    """Draw dataset labels beside model predictions on a validation frame."""
    preferred = next((i for i, r in enumerate(ds.records) if any(b[0] == 1 for b in r[2])), 0)
    img, boxes, path = ds[preferred]
    gray = (img[0].numpy() * 255).astype(np.uint8)
    base = Image.fromarray(gray, 'L').convert('RGB')
    w, h = base.size
    truth = base.copy(); pred_im = base.copy()
    d = ImageDraw.Draw(truth)
    for c, x, y, bw, bh in boxes:
        d.rectangle(((x - bw / 2) * w, (y - bh / 2) * h,
                     (x + bw / 2) * w, (y + bh / 2) * h), outline=COLORS[int(c)], width=2)
    pred_boxes = infer(model, img, device, threshold)
    d = ImageDraw.Draw(pred_im)
    for c, score, (x1, y1, x2, y2) in pred_boxes:
        d.rectangle((x1*w, y1*h, x2*w, y2*h), outline=COLORS[c], width=2)
        d.text((max(0, x1*w), max(0, y1*h - 11)), f"{CLASSES[c]} {score:.2f}", fill=COLORS[c])
    canvas = Image.new('RGB', (w * 2, h + 28), '#141925')
    canvas.paste(truth, (0, 28)); canvas.paste(pred_im, (w, 28))
    label = ImageDraw.Draw(canvas)
    label.text((8, 8), 'BIRDSAI labels (validation)', fill='white')
    label.text((w + 8, 8), 'Custom ThermalScout predictions', fill='white')
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return len(pred_boxes), Path(path).name

