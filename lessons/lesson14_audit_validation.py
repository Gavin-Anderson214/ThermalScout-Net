
from pathlib import Path
from collections import Counter

project = Path(__file__).resolve().parents[1]
pilot = project / "data" / "bambi_pilot"

images_dir = pilot / "images" / "val"
labels_dir = pilot / "labels" / "val"

images = sorted(images_dir.glob("*.png"))
labels = sorted(labels_dir.glob("*.txt"))

flight_counts = Counter()

for image in images:
    name = image.stem

    if name.startswith("flight6_"):
        flight_counts[6] += 1
    elif name.startswith("flight22_"):
        flight_counts[22] += 1
    else:
        flight_counts["other"] += 1

image_names = {p.stem for p in images}
label_names = {p.stem for p in labels}

print("Total images:", len(images))
print("Total labels:", len(labels))

print("\nImages by flight:")
for flight, count in sorted(
    flight_counts.items(), key=lambda item: str(item[0])
):
    print(f"Flight {flight}: {count}")

print("\nImages missing labels:")
print(sorted(image_names - label_names))

print("\nLabels missing images:")
print(sorted(label_names - image_names))

print("\nFirst 10 validation images:")
for image in images[:10]:
    print(image.name)

