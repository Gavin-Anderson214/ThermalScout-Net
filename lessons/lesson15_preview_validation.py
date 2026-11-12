
from pathlib import Path
import cv2

project = Path(__file__).resolve().parents[1]
pilot = project / "data" / "bambi_pilot"

images_dir = pilot / "images" / "val"
labels_dir = pilot / "labels" / "val"
preview_dir = pilot / "preview_val"

preview_dir.mkdir(parents=True, exist_ok=True)

images = sorted(images_dir.glob("*.png"))

# Select a few examples from each validation flight.
selected = []

for flight in (6, 22):
    flight_images = [
        p for p in images
        if p.name.startswith(f"flight{flight}_")
    ]
    selected.extend(flight_images[:3])

for image_path in selected:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read {image_path}")

    label_path = labels_dir / (image_path.stem + ".txt")

    height, width = image.shape[:2]

    for line in label_path.read_text().splitlines():
        class_id, cx, cy, bw, bh = line.split()

        class_id = int(class_id)
        cx, cy, bw, bh = map(float, (cx, cy, bw, bh))

        x1 = round((cx - bw / 2) * width)
        y1 = round((cy - bh / 2) * height)
        x2 = round((cx + bw / 2) * width)
        y2 = round((cy + bh / 2) * height)

        name = "Human" if class_id == 1 else "Animal"
        color = (255, 255, 0) if class_id == 1 else (0, 255, 0)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image, name, (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
        )

    output = preview_dir / image_path.name
    cv2.imwrite(str(output), image)
    print("Saved:", output.name)

print("\nValidation previews:", preview_dir)
