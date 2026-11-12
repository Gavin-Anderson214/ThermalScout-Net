
from pathlib import Path
from collections import Counter
import shutil

project = Path(__file__).resolve().parents[1]
pilot = project / "data" / "bambi_pilot"

# Existing prepared training files
source_images = pilot / "images_512"
source_labels = pilot / "labels_preview"

# Final training directories
train_images = pilot / "images" / "train"
train_labels = pilot / "labels" / "train"

train_images.mkdir(parents=True, exist_ok=True)
train_labels.mkdir(parents=True, exist_ok=True)

# Copy only the 30 prepared training pairs.
copied = 0

for image in sorted(source_images.glob("*.png")):
    label = source_labels / (image.stem + ".txt")

    if not label.exists():
        raise FileNotFoundError(label)

    shutil.copy2(image, train_images / image.name)
    shutil.copy2(label, train_labels / label.name)

    copied += 1

print("Training image-label pairs:", copied)

# Count both classes in each split.
for split in ("train", "val"):
    images_dir = pilot / "images" / split
    labels_dir = pilot / "labels" / split

    counts = Counter()
    image_count = 0

    for image in sorted(images_dir.glob("*.png")):
        label_path = labels_dir / (image.stem + ".txt")

        if not label_path.exists():
            raise FileNotFoundError(label_path)

        image_count += 1

        for line in label_path.read_text().splitlines():
            parts = line.split()

            if len(parts) != 5:
                raise ValueError(f"Bad label: {label_path}")

            class_id = int(parts[0])

            if class_id not in (0, 1):
                raise ValueError(f"Invalid class: {class_id}")

            counts[class_id] += 1

    print(f"\n{split.upper()} DATASET")
    print("Images:", image_count)
    print("Animal boxes:", counts[0])
    print("Human boxes:", counts[1])

    if counts[0] == 0 or counts[1] == 0:
        print("WARNING: One class is missing!")

print("\nDataset folder:", pilot)

