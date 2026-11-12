
from pathlib import Path
import cv2

# Locate the ThermalScout-Net project.
project = Path(__file__).resolve().parents[1]
pilot = project / "data" / "bambi_pilot"

originals = pilot / "original_thermal"
labels = pilot / "labels_preview"

# Save 512x512 images separately.
output = pilot / "images_512"
output.mkdir(parents=True, exist_ok=True)

preview_dir = pilot / "preview_512"
preview_dir.mkdir(parents=True, exist_ok=True)

size = 512
saved = 0

for image_path in sorted(originals.glob("*.png")):

    # Read the original thermal frame.
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(
            f"Cannot read {image_path.name}"
        )

    # Resize without cropping.
    resized = cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_AREA
    )

    # Keep the training image free of drawn boxes.
    cv2.imwrite(
        str(output / image_path.name),
        resized
    )

    # Make a separate visualization.
    preview = cv2.cvtColor(
        resized,
        cv2.COLOR_GRAY2BGR
    )

    label_path = labels / (
        image_path.stem + ".txt"
    )

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    for line in label_path.read_text().splitlines():

        parts = line.split()
        class_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])

        # Convert normalized box back to pixels.
        x1 = round((cx - w / 2) * size)
        y1 = round((cy - h / 2) * size)
        x2 = round((cx + w / 2) * size)
        y2 = round((cy + h / 2) * size)

        # Human = cyan, Animal = green.
        if class_id == 1:
            name = "Human"
            color = (255, 255, 0)
        else:
            name = "Animal"
            color = (0, 255, 0)

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            preview,
            name,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )

    cv2.imwrite(
        str(preview_dir / image_path.name),
        preview
    )

    saved += 1

print("Images prepared:", saved)
print("Training image size:", size, "x", size)
print("Training images:", output)
print("Labeled previews:", preview_dir)
print("Original images preserved.")
