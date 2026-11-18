"""Make matched-size MONET and Wageningen training mixes; preserve BIRDSAI validation."""

import argparse
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random
from zipfile import ZipFile

import numpy as np
from PIL import Image

from scripts.prepare_hit_uav_mix import write_mix
from thermalscout.data import BirdsaiDataset


def evenly_spaced(items, count):
    if count > len(items):
        raise ValueError(f"Need {count} frames, found {len(items)}")
    return [items[i] for i in np.linspace(0, len(items) - 1, count, dtype=int)]


def yolo_row(cls, x1, y1, x2, y2, width, height):
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Invalid box extent")
    values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
              (x2 - x1) / width, (y2 - y1) / height)
    return str(cls) + " " + " ".join(f"{v:.8f}" for v in values)


def monet_frames(archives, count):
    candidates = []
    for path in archives:
        with ZipFile(path) as archive:
            prefix = path.stem + "/"
            annotations = json.loads(archive.read(prefix + "annotations/labels.json"))
            categories = {item["id"]: item["name"] for item in annotations["categories"]}
            by_image = defaultdict(list)
            for item in annotations["annotations"]:
                by_image[item["image_id"]].append(item)
            for image in annotations["images"]:
                objects = by_image[image["id"]]
                # Do not teach unlabelled vehicles or exclusion areas as background.
                if not objects or any(categories[obj["category_id"]] != "person" for obj in objects):
                    continue
                rows = []
                for obj in objects:
                    x, y, w, h = obj["bbox"]
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(image["width"], x + w), min(image["height"], y + h)
                    if x2 > x1 and y2 > y1:
                        rows.append(yolo_row(1, x1, y1, x2, y2, image["width"], image["height"]))
                if rows:
                    candidates.append((path, prefix + "lwir/" + image["file_name"], rows))
    candidates.sort(key=lambda entry: (entry[0].stem, entry[1]))
    selected = evenly_spaced(candidates, count)
    frames = []
    for path, member, rows in selected:
        with ZipFile(path) as archive:
            content = archive.read(member)
        with Image.open(BytesIO(content)) as image:
            image.verify()
        frames.append((f"monet_{path.stem}_{Path(member).name}", content, rows))
    return frames, len(candidates), [f"{p.name}:{m}" for p, m, _ in selected]


def cow_frames(path, count):
    prefix = "COW_MOTS dataset/train/"
    with ZipFile(path) as archive:
        groups = defaultdict(list)
        for name in archive.namelist():
            if name.startswith(prefix + "images/") and name.endswith(".png"):
                group = name.removeprefix(prefix + "images/").split("/")[0]
                groups[group].append(name)
        for group in groups:
            groups[group].sort()
        # Equal coverage of all five official training videos.
        selected = []
        for index, group in enumerate(sorted(groups)):
            n = count // len(groups) + (index < count % len(groups))
            selected.extend(evenly_spaced(groups[group], n))
        frames = []
        for name in selected:
            content = archive.read(name)
            mask_name = name.replace("/images/", "/instances/", 1)
            with Image.open(BytesIO(content)) as image, Image.open(BytesIO(archive.read(mask_name))) as mask:
                width, height = image.size
                values = np.asarray(mask)
                if values.shape != (height, width):
                    raise ValueError(f"Image/mask size mismatch: {name}")
                ids = np.unique(values)
                ids = ids[ids != 0]
                if not len(ids) or np.any(ids // 1000 != 1):
                    raise ValueError(f"Unexpected cow instance codes: {mask_name}")
                rows = []
                for instance_id in ids:
                    ys, xs = np.where(values == instance_id)
                    rows.append(yolo_row(0, int(xs.min()), int(ys.min()),
                                         int(xs.max()) + 1, int(ys.max()) + 1, width, height))
            frames.append((f"wageningen_{name.removeprefix(prefix + 'images/').replace('/', '_')}", content, rows))
    return frames, sum(map(len, groups.values())), selected


def replacements(dataset, cls, count):
    groups = defaultdict(list)
    for image, _, boxes in dataset.records:
        if {box[0] for box in boxes} == {cls}:
            groups[image.stem.split("__f", 1)[0]].append(image.name)
    eligible = [name for group in sorted(groups) for name in sorted(groups[group])[1:]]
    if count > len(eligible):
        raise ValueError(f"Only {len(eligible)} replaceable class {cls} frames")
    return set(random.Random(42 + cls).sample(eligible, count))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/processed/yolo_real"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--monet-count", type=int, default=0)
    parser.add_argument("--cow-count", type=int, default=0)
    parser.add_argument("--monet-archive", type=Path, nargs="+", default=[Path("data/sources/monet/runway_01.zip")])
    parser.add_argument("--cow-archive", type=Path, default=Path("data/sources/wageningen_cow/COW_MOTS_dataset.zip"))
    args = parser.parse_args()
    if args.out.exists() or min(args.monet_count, args.cow_count) < 0 or not (args.monet_count or args.cow_count):
        parser.error("Use a fresh output folder and positive source counts")
    train = BirdsaiDataset(args.base, "train", 256, 0, 42)
    val = BirdsaiDataset(args.base, "val", 256, 0, 42)
    replaced = set()
    external = []
    metadata = {"source_partition": "official training only", "monet_url": "https://tev.fbk.eu/resources/monet",
                "wageningen_url": "https://zenodo.org/records/6370315", "licenses": "CC BY 4.0"}
    if args.monet_count:
        frames, eligible, selected = monet_frames(args.monet_archive, args.monet_count)
        external.extend(frames)
        replaced.update(replacements(train, 1, args.monet_count))
        metadata.update(monet_eligible=eligible, monet_selected=selected,
                        monet_archives={p.name: sha256(p.read_bytes()).hexdigest() for p in args.monet_archive})
    if args.cow_count:
        frames, eligible, selected = cow_frames(args.cow_archive, args.cow_count)
        external.extend(frames)
        replaced.update(replacements(train, 0, args.cow_count))
        metadata.update(wageningen_eligible=eligible, wageningen_selected=selected,
                        wageningen_archive_sha256=sha256(args.cow_archive.read_bytes()).hexdigest(),
                        wageningen_note="False-color thermal overlay; model reads its grayscale conversion")
    write_mix(args.base, args.out, train, val, replaced, external, metadata)


if __name__ == "__main__":
    main()
