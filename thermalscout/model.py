# ThermalScout-Net | COSC 2345 Introduction to AI
# Each module has one job. The network learns from annotated thermal images.
# IMPORTANT: TestReal is not used during training or development evaluation.

"""Neural network structure: learn image features, combine them, predict boxes."""

import torch
from torch import nn
import torch.nn.functional as F

# These are the class indices in the prepared BIRDSAI dataset.
CLASSES = ("animal", "human")  # 0 = animal, 1 = human
COLORS = ("#A8FF78", "#53C6FF")

def conv(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """Build one small convolution block: filter pixels, normalize, then activate."""
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

class ThermalScoutReal(nn.Module):
    """Original small detector: stride-4 fusion of shallow and deep features.

    Each cell outputs [objectness, x-offset, y-offset, log-width, log-height,
    animal-class-logit, human-class-logit]. Predicted box center belongs to its
    cell; width and height are measured in units of grid cells.
    """
    def __init__(self):
        super().__init__()
        self.low = nn.Sequential(conv(1, 16, 2), conv(16, 24, 2))  # stride 4
        self.deep = nn.Sequential(conv(24, 40, 2), conv(40, 56))    # stride 8
        self.fuse = nn.Sequential(conv(24 + 56, 48), conv(48, 48))
        self.head = nn.Conv2d(48, 7, kernel_size=1)

    def forward(self, x):
        """Pass images through the CNN to get prediction maps."""
        low = self.low(x)
        deep = self.deep(low)
        deep = F.interpolate(deep, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.fuse(torch.cat((low, deep), dim=1)))

