
import torch

# A tiny 4x4 grayscale thermal example
pixels = torch.tensor([
    [20, 35, 40, 25],
    [30, 210, 230, 35],
    [45, 200, 245, 40],
    [25, 50, 55, 30]
], dtype=torch.float32)

# Normalize pixel intensities from 0-255 to 0-1
image = pixels / 255.0

# Add the channel and batch dimensions
image = image.unsqueeze(0).unsqueeze(0)

print("Tensor shape:", image.shape)
print("Bright pixel value:", image[0, 0, 2, 2].item())
print("Dark pixel value:", image[0, 0, 0, 0].item())
