"""Load detector weights only when their preprocessing and class metadata agree."""
from pathlib import Path
import torch
from .model import CLASSES, ThermalScoutReal


def load_detector(path: Path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if tuple(checkpoint.get('classes', ())) != CLASSES:
        raise ValueError(f'Checkpoint class order must be {CLASSES}')
    size = checkpoint.get('image_size')
    if not isinstance(size, int) or isinstance(size, bool) or size < 64 or size % 8:
        raise ValueError('Checkpoint image_size must be >=64 and divisible by 8')
    model = ThermalScoutReal().to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    return model, checkpoint
