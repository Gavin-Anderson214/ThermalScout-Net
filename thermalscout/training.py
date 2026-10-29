# ThermalScout-Net | COSC 2345 Introduction to AI
# Each module has one job. The network learns from annotated thermal images.
# IMPORTANT: TestReal is not used during training or development evaluation.

"""Teach the CNN to predict objects: target creation, losses, training steps."""

import math
import torch
from torch import nn
import torch.nn.functional as F

def make_targets(batch_boxes, shape, device):
    """Place each labeled object into a grid cell for the network to learn."""
    b, _, h, w = shape
    obj = torch.zeros((b, h, w), device=device)
    cls = torch.zeros((b, h, w), dtype=torch.long, device=device)
    loc = torch.zeros((b, 4, h, w), device=device)
    collisions = 0
    for i, boxes in enumerate(batch_boxes):
        for c, x, y, bw, bh in boxes:
            gx = min(w - 1, max(0, int(x * w)))
            gy = min(h - 1, max(0, int(y * h)))
            if obj[i, gy, gx] > 0:
                collisions += 1
                continue  # one box per grid cell in this prototype
            obj[i, gy, gx] = 1
            cls[i, gy, gx] = int(c)
            loc[i, :, gy, gx] = torch.tensor([
                x * w - gx, y * h - gy,
                math.log(max(1e-4, bw * w)), math.log(max(1e-4, bh * h))
            ], device=device)
    return obj, cls, loc, collisions

def criterion(pred, boxes):
    """Measure how wrong the object presence, class, and box predictions are."""
    obj, classes, target_box, collisions = make_targets(boxes, pred.shape, pred.device)
    positive = obj.bool()
    raw_obj = pred[:, 0]
    if positive.any():
        positive_obj = F.binary_cross_entropy_with_logits(raw_obj[positive], obj[positive])
        pred_box = pred[:, 1:5].clone()
        pred_box[:, :2] = torch.sigmoid(pred_box[:, :2])
        reg = F.smooth_l1_loss(pred_box.permute(0, 2, 3, 1)[positive],
                               target_box.permute(0, 2, 3, 1)[positive])
        logits = pred[:, 5:7].permute(0, 2, 3, 1)[positive]
        class_loss = F.cross_entropy(logits, classes[positive],
                                     weight=torch.tensor([1., 2.], device=pred.device))
    else:
        positive_obj = pred.sum() * 0.0
        reg = pred.sum() * 0.0
        class_loss = pred.sum() * 0.0
    negative = ~positive
    negative_obj = F.binary_cross_entropy_with_logits(raw_obj[negative], obj[negative]) \
                   if negative.any() else pred.sum() * 0.0
    total = positive_obj + 0.25 * negative_obj + 5.0 * reg + class_loss
    return total, collisions

def train_epoch(model, loader, optimizer, device):
    """Show all batches to the model and update weights using backpropagation."""
    model.train()
    cumulative, collisions = 0.0, 0
    for images, boxes, _ in loader:
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(images)
        loss, overlaps = criterion(pred, boxes)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        cumulative += loss.item()
        collisions += overlaps
    return cumulative / len(loader), collisions

@torch.no_grad()
def eval_epoch(model, loader, device):
    """Measure validation loss without changing the learned weights."""
    model.eval()
    total, collisions = 0.0, 0
    for images, boxes, _ in loader:
        pred = model(images.to(device))
        loss, overlaps = criterion(pred, boxes)
        total += loss.item()
        collisions += overlaps
    return total / len(loader), collisions

