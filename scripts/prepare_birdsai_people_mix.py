"""Replace animal-only training frames with more people from BIRDSAI TrainReal."""

import argparse
from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path

from PIL import Image

from scripts.birdsai_prepare import clip_box, frame_files, frame_map, read_annotations
from scripts.prepare_hit_uav_mix import select_replacements, write_mix
from thermalscout.data import BirdsaiDataset


def pick_spread_frames(candidates, existing, count):
    """Add frames farthest in time from the already selected frames."""
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} new human frames available for {count} slots.")
    distances = {frame: min(abs(frame - old) for old in existing) for frame in candidates}
    for _ in range(count):
        chosen = max(distances, key=lambda frame: (distances[frame], -frame))
        del distances[chosen]
        for frame, distance in distances.items():
            distances[frame] = min(distance, abs(frame - chosen))
        yield chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-real", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("data/processed/yolo_real"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    raw = args.train_real.resolve()
    if raw.name.casefold() != "trainreal" or not (raw / "annotations").is_dir():
        parser.error("Supply the original TrainReal folder.")
    if args.count < 1 or args.out.exists():
        parser.error("Use a positive replacement count and a fresh output folder.")

    train = BirdsaiDataset(args.base, "train", 256, 0, 42)
    validation = BirdsaiDataset(args.base, "val", 256, 0, 42)
    split = json.loads((args.base / "split_videos.json").read_text(encoding="utf-8"))
    originals = defaultdict(set)
    human_videos = set()
    for image, _, boxes in train.records:
        video, frame = image.stem.split("__f")
        originals[video].add(int(frame))
        if any(box[0] == 1 for box in boxes):
            human_videos.add(video)
    if not human_videos or human_videos - set(split["train_videos"]) or human_videos.intersection(split["validation_videos"]):
        raise ValueError("Extra human frames must come from existing training videos only.")
    replaced = select_replacements(train, args.count)
    per_video, remainder = divmod(args.count, len(human_videos))
    new_frames = []
    for index, video in enumerate(sorted(human_videos)):
        annotations, _ = read_annotations(raw / "annotations" / f"{video}.csv")
        paired, _, _ = frame_map(frame_files(raw / "images" / video), set(annotations), None)
        if not annotations or len(paired) / len(annotations) < 0.7:
            raise ValueError(f"Poor frame to annotation match for {video}")
        candidates = {frame for frame in paired.keys() - originals[video]
                      if any(ann.cls == 1 for ann in annotations[frame])}
        count = per_video + (index < remainder)
        for frame in pick_spread_frames(candidates, originals[video], count):
            path = paired[frame]
            content = path.read_bytes()
            with Image.open(BytesIO(content)) as image:
                image.load()
                width, height = image.size
            rows = []
            for ann in annotations[frame]:
                box = clip_box(ann, width, height)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                rows.append(f"{ann.cls} {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} "
                            f"{(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
            if not any(row.startswith("1 ") for row in rows):
                raise ValueError(f"No valid human box in {path.name}")
            new_frames.append((f"{video}__f{frame:07d}{path.suffix.lower()}", content, rows))
        print(f"Selected {count} additional human frames from {video}", flush=True)
    metadata = {
        "source": "https://lila.science/datasets/conservationdrones",
        "source_partition": "TrainReal, original training videos",
        "source_license": "Community Data License Agreement Permissive",
        "selected_birdsai_frames": [name for name, _, _ in new_frames],
    }
    write_mix(args.base, args.out, train, validation, replaced, new_frames, metadata)


if __name__ == "__main__":
    main()
