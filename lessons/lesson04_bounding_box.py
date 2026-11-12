# ThermalScout-Net
# Lesson 4: Bounding Box Decoding

# Normalized center coords and dimensions of the bounding box
cx = 0.50
cy = 0.50
w = 0.20
h = 0.40

#Convert center/size3 to corner coordinates
x1 = cx - w / 2
y1 = cy - h / 2
x2 = cx + w / 2
y2 = cy + h / 2

print("Bounding box coordinates:")
print("Top-left corner: (", x1, ",", y1, ")")
print("Bottom-right corner: (", x2, ",", y2, ")")

#Convert to pixel coordinates for a 512x512 image
image_size = 512

print("\nPixel Coordinates for a 512x512 image: ")
print("Top-left:", (round(x1 * image_size),
                    round(y1 * image_size)))

print("Bottom-right:", (round(x2 * image_size),
                       round(y2 * image_size)))

# draw ffirst bounding box

import cv2
import numpy as np

# Create a simulated grayscale thermal image
image = np.zeros((512, 512), dtype=np.uint8)

# Add a bright region representing a warm object.
cv2.ellipse(
    image,
    (256, 256),
    (45, 95),
    0, 0, 360,
    220,
    -1
)

# Convert grayscale to color so we can draw a box.
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Use the bounding-box coordinates from the lesson.
top_left = (205, 154)
bottom_right = (307, 358)

# Draw a green rectangle.
cv2.rectangle(
    image,
    top_left,
    bottom_right,
    (0, 255, 0),
    2
)

# Add a class label.
cv2.putText(
    image,
    "Human",
    (205, 145),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

# Display the result.
cv2.imshow("ThermalScout-Net Bounding Box", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
