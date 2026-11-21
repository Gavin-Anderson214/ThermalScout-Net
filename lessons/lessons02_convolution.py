
import torch
from torch import nn

# Step 1: Generate a random 4x4 thermal image.
# Values range from 0 (dark) to 1 (bright).
image = torch.rand(4, 4)

print("Original thermal image:")
print(image)

# Step 2: Add batch and channel dimensions.
image = image.unsqueeze(0).unsqueeze(0)

# Step 3: Create a 3x3 convolutional filter.
conv = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    padding=0,
    bias=False
)

# Step 4: Inspect the randomly initialized filter.
print("\nRandom filter weights:")
print(conv.weight.data[0, 0])

# Step 5: Apply convolution.
# A 3-by-3 window fits in four places on this 4-by-4 image. Each output is a weighted
# sum of nine pixels.
features = conv(image)

# Add a positive offset for this demonstration
features = features + 0.3

# Apply ReLU
relu = nn.ReLU()
activated_features = relu(features)

print("\nBefore ReLU:")
print(features[0, 0])

print("\nAfter ReLU:")
print(activated_features[0, 0])


# Step 6: Display results.
print("\nInput shape:", image.shape)
print("Filter shape:", conv.weight.shape)
print("Output shape:", features.shape)
print("\nFeature map:")
print(features[0, 0])
