
import csv
from collections import defaultdict
from pathlib import Path

import cv2

project = Path(__file__).resolve().parents[1]
source = project / "data" / "bambi_source" / "bambi_downloads"
pilot = project / "data" / "bambi_pilot"

images_out = pilot / "images" / "val"
labels_out = pilot / "labels" / "val"

images_out.mkdir(parents=True, exist_ok=True)
labels_out.mkdir(parents=True, exist_ok=True)

# These flights are reserved for validation.
flights = [6, 22]
max_frames_per_flight = 15

# Start with species known to occur in these flights.
animal_species = {
    "Cervus elaphus",
    "Sus scrofa",
    "Canis lupus familiaris",
}

for flight in flights:
    video_path = source / f"{flight}_matched_processed.mp4"
    annotations_path = source / f"{flight}_gt.txt"

    if not video_path.exists() or not annotations_path.exists():
        raise FileNotFoundError(
            f"Missing video or labels for flight {flight}"
        )

    frame_labels = defaultdict(list)
    unknown_species = set()
    unknown_frames = set()

    # Read BAMBI annotations.
    with annotations_path.open(encoding="utf-8") as file:
        for row in csv.reader(file):
            if len(row) != 13:
                continue

            frame = int(row[0])
            x, y, w, h = map(float, row[2:6])
            species = row[9].split(" (")[0].strip()

            if species == "Homo sapiens":
                class_id = 1
            elif species in animal_species:
                class_id = 0
            else:
                unknown_species.add(species)
                unknown_frames.add(frame)
                continue

            if w <= 0 or h <= 0:
                continue

            if not (0 <= x and 0 <= y and
                    x + w <= 1024 and y + h <= 1024):
                continue

            cx = (x + w / 2) / 1024
            cy = (y + h / 2) / 1024
            nw = w / 1024
            nh = h / 1024

            frame_labels[frame].append(
                f"{class_id} {cx:.6f} {cy:.6f} "
                f"{nw:.6f} {nh:.6f}"
            )

    if unknown_species:
        print(
            f"Flight {flight}: encountered unmapped species:",
            sorted(unknown_species)
        )

    frames = sorted(
    frame for frame in frame_labels
    if frame not in unknown_frames
)

    if not frames:
        raise RuntimeError(
            f"No valid annotations in flight {flight}"
        )

    # Select frames spaced across the flight.
    step = max(1, len(frames) // max_frames_per_flight)
    selected = frames[::step][:max_frames_per_flight]

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open flight {flight}")

    saved = 0

    for frame_num in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ok, frame = cap.read()

        if not ok or frame.shape[0] != 1024:
            print("Skipped:", flight, frame_num)
            continue

        # Extract the full thermal half.
        thermal = frame[:, :1024]

        # Convert to a single grayscale channel.
        gray = cv2.cvtColor(
            thermal, cv2.COLOR_BGR2GRAY
        )

        resized = cv2.resize(
            gray, (512, 512),
            interpolation=cv2.INTER_AREA
        )

        name = f"flight{flight}_frame{frame_num:06d}"

        cv2.imwrite(
            str(images_out / f"{name}.png"),
            resized
        )

        (labels_out / f"{name}.txt").write_text(
            "\n".join(frame_labels[frame_num]) + "\n"
        )

        saved += 1

    cap.release()
    print(f"Flight {flight}: saved {saved} validation images")

print("\nValidation images:", images_out)
print("Validation labels:", labels_out)
