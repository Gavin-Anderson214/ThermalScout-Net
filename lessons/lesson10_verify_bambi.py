
from pathlib import Path
from collections import Counter
import cv2

# Locate project folders.
project = Path(__file__).resolve().parents[1]
pilot = project / "data" / "bambi_pilot"

images_dir = pilot / "original_thermal"
labels_dir = pilot / "labels_preview"

images = sorted(images_dir.glob("*.png"))

if not images:
    raise RuntimeError("No extracted thermal images found.")

counts = Counter()
errors = []

for image_path in images:
    # Find the corresponding YOLO label.
    label_path = labels_dir / (image_path.stem + ".txt")

    if not label_path.exists():
        errors.append(f"Missing label: {image_path.name}")
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        errors.append(f"Unreadable image: {image_path.name}")
        continue

    if image.shape[:2] != (1024, 1024):
        errors.append(f"Wrong size: {image_path.name}")
        continue

    # Read every bounding-box label.
    for line in label_path.read_text().splitlines():
        parts = line.split()

        if len(parts) != 5:
            errors.append(f"Bad label: {label_path.name}")
            continue

        class_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])

        if class_id not in (0, 1):
            errors.append(f"Unknown class: {label_path.name}")

        if not (0 <= cx <= 1 and 0 <= cy <= 1
                and 0 < w <= 1 and 0 < h <= 1):
            errors.append(f"Invalid box: {label_path.name}")

        counts[class_id] += 1

print("Thermal images:", len(images))
print("Animal boxes:", counts[0])
print("Human boxes:", counts[1])
print("Total boxes:", sum(counts.values()))
print("Errors:", len(errors))

for error in errors[:10]:
    print("ERROR:", error)

if not errors:
    print("\nVerification passed!")
