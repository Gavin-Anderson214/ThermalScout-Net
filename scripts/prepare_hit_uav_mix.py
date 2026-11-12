"""Prepare a matched-size BIRDSAI and HIT-UAV development experiment."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from itertools import zip_longest
import json
import math
from pathlib import Path
import random
import re
from shutil import copyfile
from urllib.request import urlopen

from PIL import Image

from thermalscout.data import BirdsaiDataset


HIT_UAV_REVISION = "669259659b21a2737ca9f5e4bc81892acbe6ecd8"
HIT_UAV_REPO = "https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset"
HIT_UAV_RAW = f"https://raw.githubusercontent.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset/{HIT_UAV_REVISION}"
NIGHT_FRAME = re.compile(r"1_(\d+)_(\d+)_0_\d+\.jpg")


def cached_file(cache, name, url):
    path = cache / name
    if path.is_file():
        return path.read_bytes()
    with urlopen(url, timeout=30) as response:
        content = response.read()
    path.write_bytes(content)
    return content


def human_rows(annotations, width, height):
    rows = []
    for item in annotations:
        x, y, w, h = item["bbox"]
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError("Non-finite source box")
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                  (x2 - x1) / width, (y2 - y1) / height)
        rows.append("1 " + " ".join(f"{value:.8f}" for value in values))
    return rows


def select_external(metadata, count, rng):
    by_image = defaultdict(list)
    for annotation in metadata["annotation"]:
        by_image[annotation["image_id"]].append(annotation)
    categories = {category["name"]: category["id"] for category in metadata["categories"]}
    groups = defaultdict(list)
    for image in metadata["images"]:
        name = image["filename"]
        match = NIGHT_FRAME.fullmatch(name)
        if match is None:
            continue
        annotations = by_image[image["id"]]
        if any(a["category_id"] == categories["DontCare"] or a.get("iscrowd") for a in annotations):
            continue
        humans = [a for a in annotations if a["category_id"] == categories["Person"]]
        rows = human_rows(humans, image["width"], image["height"])
        if rows:
            group = (image["date_captured"], *match.groups())
            groups[group].append((name, image["width"], image["height"], rows))
    buckets = [sorted(groups[group]) for group in sorted(groups)]
    for bucket in buckets:
        rng.shuffle(bucket)
    candidates = [item for group in zip_longest(*buckets) for item in group if item is not None]
    if len(candidates) < count:
        raise ValueError(f"Requested {count} frames, but only {len(candidates)} matched")
    return candidates[:count], len(candidates)


def digest_split(root, split):
    digest = sha256()
    for category in ("images", "labels"):
        for path in sorted((root / category / split).iterdir()):
            if path.is_file():
                digest.update(f"{category}/{split}/{path.name}\n".encode())
                digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def select_replacements(train, count):
    eligible = defaultdict(list)
    for image, _, boxes in train.records:
        if {box[0] for box in boxes} == {0}:
            eligible[image.stem.split("__f", 1)[0]].append(image.name)
    # Retain at least one animal-only frame from every training video that has one.
    removable = [name for video in sorted(eligible) for name in eligible[video][1:]]
    if len(removable) < count:
        raise ValueError(f"Only {len(removable)} animal-only frames can be replaced.")
    return set(random.Random(42).sample(removable, count))


def write_mix(base, out, train, validation, replaced, new_frames, metadata):
    names = {image.name for image, _, _ in train.records}
    new_names = [name for name, _, _ in new_frames]
    if len(set(new_names)) != len(replaced) or names.intersection(new_names):
        raise ValueError("Each replaced frame needs one distinct new image.")
    out.mkdir(parents=True, exist_ok=False)
    for category in ("images", "labels"):
        for split in ("train", "val"):
            (out / category / split).mkdir(parents=True)
    for split, dataset in (("train", train), ("val", validation)):
        for image, label, _ in dataset.records:
            if split == "train" and image.name in replaced:
                continue
            copyfile(image, out / "images" / split / image.name)
            copyfile(label, out / "labels" / split / label.name)
    for name, content, rows in new_frames:
        (out / "images" / "train" / name).write_bytes(content)
        (out / "labels" / "train" / f"{Path(name).stem}.txt").write_text(
            "\n".join(rows) + "\n", encoding="utf-8")
    output_path = out.resolve().as_posix().replace("'", "''")
    (out / "data.yaml").write_text(
        f"path: '{output_path}'\ntrain: images/train\nval: images/val\nnc: 2\nnames: [animal, human]\n",
        encoding="utf-8",
    )
    mixed = BirdsaiDataset(out, "train", 256, 0, 42)
    val_digest = digest_split(base, "val")
    if len(mixed) != len(train) or digest_split(out, "val") != val_digest:
        raise RuntimeError("Training count or validation bytes differ from the control.")
    summary = {
        **metadata,
        "sampling_seed": 42,
        "training_images": len(mixed),
        "validation_images": len(validation),
        "original_train_boxes": dict(zip(("animal", "human"), train.label_counts)),
        "mixed_train_boxes": dict(zip(("animal", "human"), mixed.label_counts)),
        "original_train_sha256": digest_split(base, "train"),
        "mixed_train_sha256": digest_split(out, "train"),
        "validation_sha256": val_digest,
        "replaced_birdsai_frames": sorted(replaced),
    }
    (out / "dataset_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Train: {len(mixed)} frames, {mixed.label_counts[0]} animal boxes, {mixed.label_counts[1]} human boxes")
    print(f"Validation: {len(validation)} original frames, verified byte-for-byte")
    print(f"Dataset ready at {out.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/processed/yolo_real"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200, help="Number of training images to replace")
    args = parser.parse_args()
    if args.count < 1 or args.out.exists():
        parser.error("Use a positive replacement count and a fresh output folder.")

    train = BirdsaiDataset(args.base, "train", 256, 0, 42)
    validation = BirdsaiDataset(args.base, "val", 256, 0, 42)
    replaced = select_replacements(train, args.count)
    cache = Path("data/sources") / f"hit_uav_{HIT_UAV_REVISION[:7]}"
    cache.mkdir(parents=True, exist_ok=True)
    annotation_bytes = cached_file(cache, "train.json", f"{HIT_UAV_RAW}/normal_json/annotations/train.json")
    cached_file(cache, "LICENSE", f"{HIT_UAV_RAW}/LICENSE")
    selected, candidate_count = select_external(json.loads(annotation_bytes), args.count, random.Random(42))

    def load_frame(frame):
        name, width, height, rows = frame
        content = cached_file(cache, name, f"{HIT_UAV_RAW}/normal_json/train/{name}")
        with Image.open(BytesIO(content)) as image:
            if image.format != "JPEG" or image.size != (width, height):
                raise ValueError(f"Image does not match source annotations: {name}")
            image.load()
        return f"hit_uav_{name}", content, rows

    print(f"Preparing {len(selected)} night images from the HIT-UAV training partition.", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        external = list(pool.map(load_frame, selected))

    metadata = {
        "source": HIT_UAV_REPO,
        "source_revision": HIT_UAV_REVISION,
        "source_license": "CC BY 4.0",
        "source_paper": "https://doi.org/10.1038/s41597-023-02066-6",
        "source_annotation_sha256": sha256(annotation_bytes).hexdigest(),
        "source_partition": "normal_json/train",
        "eligible_external_frames": candidate_count,
        "selected_hit_uav_frames": [name for name, _, _, _ in selected],
    }
    write_mix(args.base, args.out, train, validation, replaced, external, metadata)


if __name__ == "__main__":
    main()
