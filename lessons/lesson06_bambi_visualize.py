import csv
from pathlib import Path
import cv2

# Find the main ThermalScout-Net project.
project = Path(__file__).resolve().parents[1]
bambi = project / "data" / "bambi_source"

# Locate the downloaded video and annotations.
videos = list(bambi.rglob("47_matched_processed.mp4"))
labels = list(bambi.rglob("47_gt.txt"))

if not videos or not labels:
    raise FileNotFoundError(
        "Flight 47 video or annotations not found."
    )

# Open flight 47.
cap = cv2.VideoCapture(str(videos[0]))

if not cap.isOpened():
    raise RuntimeError("Could not open flight 47 video")

# Examine frame 177, which contains two humans.
frame_number = 177
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read frame 177")

# Keep the full 1024x1024 thermal half.
thermal = frame[:, :1024].copy()

# Preserve the original processed thermal frame.
output = project / "lessons" / "bambi_preview"
output.mkdir(exist_ok=True)

cv2.imwrite(
    str(output / "flight47_thermal_original.png"),
    thermal
)

# Read the labels for this frame.
with labels[0].open(encoding="utf-8") as file:
    for row in csv.reader(file):
        if len(row) != 13:
            continue

        if int(row[0]) != frame_number:
            continue

        x = int(row[2])
        y = int(row[3])
        w = int(row[4])
        h = int(row[5])

        species = row[9]

        if species.startswith("Homo sapiens"):
            name = "Human"
            color = (255, 255, 0)
        elif species.startswith("Canis lupus familiaris"):
            name = "Animal"
            color = (0, 255, 0)
        elif species.startswith("Sus scrofa"):
            name = "Animal"
            color = (0, 255, 0)
        else:
            continue

        cv2.rectangle(
            thermal, (x, y), (x+w, y+h),
            color, 2
        )

        cv2.putText(
            thermal, name, (x, max(15, y-5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 1
        )

        print("Found:", name, "at", (x, y, w, h))

# Save the annotated thermal image.
preview = output / "flight47_annotated.png"
cv2.imwrite(str(preview), thermal)

print("Saved:", preview)
print("Frame dimensions:", thermal.shape)

cv2.imshow("BAMBI Flight 47", thermal)
cv2.waitKey(0)
cv2.destroyAllWindows()
