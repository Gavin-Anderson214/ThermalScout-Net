import csv
from pathlib import Path
from collections import defaultdict

# Find the main ThermalScout-Net project folder.
project = Path(__file__).resolve().parents[1]

# Locate BAMBI's original annotation file.
bambi = project / "data" / "bambi_source"
files = list(bambi.rglob("47_gt.txt"))

if not files:
    raise FileNotFoundError("Flight 47 annotations not found")

annotation_file = files[0]

# Save converted labels separately.
output = project / "data" / "bambi_pilot" / "labels_preview"
output.mkdir(parents=True, exist_ok=True)

image_size = 1024
frame_labels = defaultdict(list)

valid_animals = (
    "Sus scrofa",
    "Canis lupus familiaris",
    "Cervus elaphus",
    "Capreolus capreolus",
    "Dama dama",
    "Capra ibex",
    "Rupicapra rupicapra",
    "Aves",
)

with annotation_file.open(encoding="utf-8") as file:
    for row in csv.reader(file):
        if len(row) != 13:
            continue

        frame = int(row[0])
        x = float(row[2])
        y = float(row[3])
        w = float(row[4])
        h = float(row[5])
        species = row[9].strip()

        # Map BAMBI species to our two classes.
        if species.startswith("Homo sapiens"):
            class_id = 1
        elif species.startswith(valid_animals):
            class_id = 0
        else:
            continue

        # Reject invalid bounding boxes.
        if w <= 0 or h <= 0:
            continue

        # Normalize the bounding-box coordinates.
        cx = (x + w / 2) / image_size
        cy = (y + h / 2) / image_size
        nw = w / image_size
        nh = h / image_size

        # Reject boxes that extend outside the image.
        if not (0 <= x and 0 <= y and
                x + w <= image_size and
                y + h <= image_size):
            continue

        label = f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        frame_labels[frame].append(label)

# Write one YOLO label file per annotated frame.
for frame, labels in frame_labels.items():
    label_path = output / f"flight47_frame{frame:06d}.txt"
    label_path.write_text("\n".join(labels) + "\n")

print("Source:", annotation_file)
print("Output:", output)
print("Annotated frames converted:", len(frame_labels))
print("Total bounding boxes:", sum(map(len, frame_labels.values())))
print("Frame 177 labels:")
for label in frame_labels.get(177, []):
    print(label)