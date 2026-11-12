from pathlib import Path
import cv2

# Locate the project and BAMBI files.
project = Path(__file__).resolve().parents[1]
bambi = project / "data" / "bambi_source"

videos = list(bambi.rglob("47_matched_processed.mp4"))

if not videos:
    raise FileNotFoundError("Flight 47 video not found")

video = videos[0]

# Read the converted label filenames.
labels_dir = project / "data" / "bambi_pilot" / "labels_preview"
label_files = sorted(labels_dir.glob("flight47_frame*.txt"))

# Select at most 30 annotated frames, spaced across the list.
if not label_files:
    raise RuntimeError("No converted labels found")

step = max(1, len(label_files) // 30)
selected = label_files[::step][:30]

# Save original-resolution thermal frames separately.
output = project / "data" / "bambi_pilot" / "original_thermal"
output.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video))

if not cap.isOpened():
    raise RuntimeError("Could not open BAMBI video")

saved = 0

for label_file in selected:
    frame_number = int(label_file.stem.split("frame")[-1])

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()

    if not success:
        print("Skipped frame:", frame_number)
        continue

    # Extract the complete thermal half.
    thermal = frame[:, :1024]

    if thermal.shape[:2] != (1024, 1024):
        print("Unexpected dimensions:", thermal.shape)
        continue

    name = f"flight47_frame{frame_number:06d}.png"

    cv2.imwrite(str(output / name), thermal)
    saved += 1

cap.release()

print("Source video:", video)
print("Selected frames:", len(selected))
print("Saved thermal frames:", saved)
print("Output folder:", output)