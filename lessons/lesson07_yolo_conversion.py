# Lesson 11: BAMBI bounding box to YOLO format

image_size = 1024

# Human bounding box from BAMBI frame 177
x = 877
y = 91
width = 37
height = 35

# Convert top-left coordinates to center coordinates
center_x = (x + width / 2) / image_size
center_y = (y + height / 2) / image_size

# Normalize width and height
norm_width = width / image_size
norm_height = height / image_size

# ThermalScout-Net: animal = 0, human = 1
class_id = 1

print("Original bounding box:")
print(x, y, width, height)

print("\nYOLO label:")
print(
    class_id,
    round(center_x, 6),
    round(center_y, 6),
    round(norm_width, 6),
    round(norm_height, 6)
)