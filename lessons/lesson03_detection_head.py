import torch
from torch import nn

# Create a simulated thermal feature map
# one image , 48 feature channels, 4x4 grid.
features = torch.rand(1, 48, 4, 4)

# ThermalScout-Net final detection layer
head = nn.Conv2d(
    in_channels=48,
    out_channels=7,  # 7 predictions per grid cell
    kernel_size=1
)

# produce seven predictions at eachg grid cell (4x4) for a total of 112 predictions
predictions = head(features)

print("Feature shape: ", features.shape)
print("Prediction shape: ", predictions.shape)

# INspect the outputs at one grid cell
cell = predictions[0, :, 0, 0]  # first grid cell (top-left)

print("\nRaw predictions:") # raw predictions for the first grid cell
print("Objectness logit:", cell[0].item()) # objectness logit
print("X offset logit:", cell[1].item()) # X offset logit
print("Y offset logit:", cell[2].item()) # Y offset logit
print("Log width:", cell[3].item()) # Log width
print("Log height:", cell[4].item()) # Log height
print("Animal logit:", cell[5].item()) # Animal logit
print("Human logit:", cell[6].item()) # Human logit

objectness = torch.sigmoid(cell[0])  # Apply sigmoid to objectness logit

# Convert class logits to probabilities using softmax
classes = torch.softmax(cell[5:7], dim=0)  # Apply softmax to animal and human logits

print("\nObject confidence: ", objectness.item())
print("Animal probabilities: ", classes[0].item())
print("Human probabilities: ", classes[1].item())

# Decode one grid cell's predictions into bounding box coordinates
# This lesson uses a simplified 4x4 grid

grid_x = 0
grid_y = 0
grid_size = 4

# Sigmoid keeps the predicted center inside the grid cell
x_offset = torch.sigmoid(cell[1])  # X offset
y_offset = torch.sigmoid(cell[2])  # Y offset

#Exponentiate the log width and height to get the actual width and height
width = torch.exp(cell[3]).item() / grid_size  # Width
height = torch.exp(cell[4]).item() / grid_size  # Height

# Convert to normalized image Coordinates (0 to 1)
center_x = (grid_x + x_offset.item()) / grid_size
center_y = (grid_y + y_offset.item()) / grid_size

x1 = center_x - width / 2
y1 = center_y - height / 2
x2 = center_x + width / 2
y2 = center_y + height / 2

print("\nDecoded bounding box coordinates:")
print("Center: ", (center_x, center_y))
print("Width: ", width)
print("Height: ", height)
print("Corners: ", (x1, y1, x2, y2))