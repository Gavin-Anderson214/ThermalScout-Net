# ThermalScout-Net | COSC 2345 Introduction to AI
# Each module has one job. The network learns from annotated thermal images.
# IMPORTANT: TestReal is not used during training or development evaluation.

"""Teach the CNN to predict objects: target creation, losses, training steps."""

import math
import torch
from torch import nn
import torch.nn.functional as F

# --- Assign each object to its center cell; count collisions because this model can learn only one box per cell.
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

# --- Combine objectness, box regression and class losses; empty images still teach background rejection.
def criterion(pred, boxes, objectness="baseline", human_weight=2.0):
    """Measure how wrong the object presence, class, and box predictions are."""
    if not math.isfinite(human_weight) or human_weight <= 0:
        raise ValueError("human_weight must be finite and positive")
    obj, classes, target_box, collisions = make_targets(boxes, pred.shape, pred.device)
    positive = obj.bool()
    raw_obj = pred[:, 0]
    if positive.any():
        # Regress boxes and classify objects only where a target is present.
        positive_obj = F.binary_cross_entropy_with_logits(raw_obj[positive], obj[positive])
        pred_box = pred[:, 1:5].clone()
        pred_box[:, :2] = torch.sigmoid(pred_box[:, :2])
        reg = F.smooth_l1_loss(pred_box.permute(0, 2, 3, 1)[positive],
                               target_box.permute(0, 2, 3, 1)[positive])
        logits = pred[:, 5:7].permute(0, 2, 3, 1)[positive]
        class_loss = F.cross_entropy(logits, classes[positive],
                                     weight=torch.tensor([1., human_weight], device=pred.device))
    else:
        positive_obj = pred.sum() * 0.0
        reg = pred.sum() * 0.0
        class_loss = pred.sum() * 0.0
    negative = ~positive
    negative_obj = F.binary_cross_entropy_with_logits(raw_obj[negative], obj[negative]) \
                   if negative.any() else pred.sum() * 0.0
    # Background loss is downweighted; localization loss receives extra weight.
    if objectness == "focal":
        # Emphasize confident false positives while reducing easy background loss.
        negative_obj = (F.binary_cross_entropy_with_logits(raw_obj[negative], obj[negative], reduction="none") * raw_obj[negative].sigmoid().pow(2)).mean() if negative.any() else pred.sum() * 0.0
    elif objectness != "baseline":
        raise ValueError(f"Unknown objectness loss: {objectness}")
    total = positive_obj + 0.25 * negative_obj + 5.0 * reg + class_loss
    return total, collisions

# --- Enable training behavior, clear old gradients, backpropagate loss and update model weights batch by batch.
def train_epoch(model, loader, optimizer, device, objectness="baseline", human_weight=2.0):
    """Show all batches to the model and update weights using backpropagation."""
    model.train()
    cumulative, collisions = 0.0, 0
    for images, boxes, _ in loader:
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(images)
        loss, overlaps = criterion(pred, boxes, objectness, human_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        cumulative += loss.item()
        collisions += overlaps
    return cumulative / len(loader), collisions

# --- Measure validation loss without gradients or weight updates; this is development feedback.
@torch.no_grad()
def eval_epoch(model, loader, device, objectness="baseline", human_weight=2.0):
    """Measure validation loss without changing the learned weights."""
    model.eval()
    total, collisions = 0.0, 0
    for images, boxes, _ in loader:
        pred = model(images.to(device))
        loss, overlaps = criterion(pred, boxes, objectness, human_weight)
        total += loss.item()
        collisions += overlaps
    return total / len(loader), collisions

