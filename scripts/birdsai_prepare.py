"""ThermalScout-Net Step 2: inspect real BIRDSAI labels, preview, prepare train/val.

Expected dataset: BIRDSAI Conservation Drones, TrainReal/annotations/*.csv and
TrainReal/images/<video>/*.jpg. Original TestReal is never read or exported.

Usage:
 python birdsai_prepare.py --data "D:\\path\\to\\BIRDSAI" --mode inspect
 python birdsai_prepare.py --data "D:\\path\\to\\BIRDSAI" --mode prepare

Requires: Pillow, matplotlib (already installed with the ThermalScout demo).
This script does not train a neural network.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise SystemExit("Pillow is missing. Run: python3.13 -m pip install pillow")

CLASS_NAMES = {0: "animal", 1: "human"}
PALETTE = {0: "#b3ff79", 1: "#55c8ff"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Annotation:
    """One labeled thermal object in one video frame."""
    frame: int
    object_id: int
    x: float
    y: float
    w: float
    h: float
    cls: int
    species: str = ""
    occlusion: str = ""
    noise: str = ""


def numeric_tail(path: Path):
    """Return the final run of digits in a stem, e.g. seq_000005.jpg -> 5."""
    m = re.search(r"(\d+)$", path.stem)
    return int(m.group(1)) if m else None


def find_train_real(root: Path) -> Path | None:
    """Locate the folder with original thermal images and CSV labels."""
    root = root.expanduser().resolve()
    if root.name.lower() == "trainreal" and root.is_dir():
        return root
    if (root / "TrainReal").is_dir():
        return root / "TrainReal"
    # Allow a dataset root with one extra level, e.g. unpacked archive folder.
    if root.is_dir():
        try:
            for child in root.iterdir():
                if child.is_dir() and (child / "TrainReal").is_dir():
                    return child / "TrainReal"
        except PermissionError:
            return None
    return None


def locate_data(path_arg: str | None) -> Path:
    """Choose the dataset location from a user path or likely folders."""
    candidates = []
    if path_arg:
        candidates = [Path(path_arg)]
    else:
        # Search around the repository; or use --data for an exact path.
        here = Path(__file__).resolve().parent
        candidates = [here, here.parent, here.parent.parent, here.parent.parent / "BIRDSAI",
                      Path(r"D:\Intro to AI project"),
                      Path(r"D:\Intro to AI project\BIRDSAI"),
                      Path(r"D:\College Work\Intro To AI\Project"),
                      Path(r"D:\College Work\Intro To AI\Project\BIRDSAI"),
                      Path(r"D:\BIRDSAI")]
    for candidate in candidates:
        found = find_train_real(candidate)
        if found and (found / "annotations").is_dir() and (found / "images").is_dir():
            return found
    checked = "\n".join(f"  - {p}" for p in candidates)
    raise SystemExit(
        "Could not locate BIRDSAI TrainReal/annotations and TrainReal/images.\n"
        "Checked:\n" + checked + "\n\n"
        "Run with an explicit dataset path, e.g.:\n"
        '  python3.13 birdsai_prepare.py --data "D:\\Your\\BIRDSAI" --mode inspect\n'
        "You may also pass the TrainReal folder directly. Make sure frames are extracted."
    )


def read_annotations(csv_path: Path):
    """Read original BIRDSAI MOT CSV: frame,id,x,y,w,h,class,species,occ,noise."""
    by_frame = defaultdict(list)
    counters = Counter()
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for line_no, row in enumerate(reader, 1):
            if not row or not any(item.strip() for item in row):
                continue
            if len(row) < 7:
                counters["invalid_columns"] += 1
                continue
            try:
                frame = int(float(row[0].strip()))
                obj_id = int(float(row[1].strip()))
                x, y, w, h = (float(s.strip()) for s in row[2:6])
                cls = int(float(row[6].strip()))
                if cls not in CLASS_NAMES or w <= 0 or h <= 0 or frame < 0:
                    counters["invalid_class_or_size"] += 1
                    continue
            except ValueError:
                counters["invalid_numeric"] += 1
                continue
            ann = Annotation(frame, obj_id, x, y, w, h, cls,
                             *(row[i].strip() if len(row) > i else "" for i in (7, 8, 9)))
            by_frame[frame].append(ann)
            counters[CLASS_NAMES[cls]] += 1
    return dict(by_frame), counters


def frame_files(seq_dir: Path):
    """Find image files in a video folder in frame order."""
    frames = [p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    frames.sort(key=lambda p: (numeric_tail(p) is None,
                               numeric_tail(p) if numeric_tail(p) is not None else 0, p.name))
    return frames


def frame_map(images: list[Path], annotation_frames: set[int], user_offset: int | None):
    """Return frame_id -> image path, selected filename offset, explanatory note.

    We do NOT guess offsets from annotation overlap alone: consecutive video
    frames can all appear to match while being displaced by one frame.
    """
    ids = [numeric_tail(p) for p in images]
    if any(x is None for x in ids) or len(set(ids)) != len(ids):
        return {}, None, "Unusable/nonunique numeric image suffixes; check filenames."
    image_by_id = dict(zip(ids, images))
    lowest_img = min(ids) if ids else None
    if user_offset is not None:
        offset = user_offset
        how = "manual --frame-offset override"
    elif lowest_img == 0:
        offset = 0
        how = "image numbering begins at 0"
    elif lowest_img == 1 and 0 in annotation_frames:
        offset = 1
        how = "image numbering begins at 1, annotations include frame 0"
    else:
        offset = 0
        how = ("default offset 0; CHECK PREVIEW: number origin is not proven. "
               "Use --frame-offset if filenames and labels are shifted")
    paired = {f: image_by_id[f + offset] for f in annotation_frames if f + offset in image_by_id}
    if annotation_frames and (len(paired) / len(annotation_frames)) < 0.70:
        note = "LOW FRAME MATCH RATE. Check CSV/image pairing and frame offset before preparing."
    else:
        note = ""
    return paired, offset, f"{how}. {note}".strip()


def locate_image_directory(images_root: Path, stem: str):
    """Find the folder holding image frames for one sequence."""
    exact = images_root / stem
    if exact.is_dir():
        return exact
    if not images_root.is_dir():
        return None
    # Case-insensitive matching on Windows; no cross-video fuzzy matches.
    for child in images_root.iterdir():
        if child.is_dir() and child.name.casefold() == stem.casefold():
            return child
    return None


def collect_sequences(train_real: Path, offset: int | None):
    """Match annotation files with image directories for each video."""
    annotations_dir = train_real / "annotations"
    images_dir = train_real / "images"
    csvs = sorted(p for p in annotations_dir.glob("*.csv") if p.is_file())
    if not csvs:
        raise SystemExit(f"No .csv files under {annotations_dir}. Is BIRDSAI extracted?")
    sequences = []
    for ann_path in csvs:
        d = locate_image_directory(images_dir, ann_path.stem)
        if d is None:
            sequences.append({"name": ann_path.stem, "error": "matching image folder missing"})
            continue
        images = frame_files(d)
        if not images:
            sequences.append({"name": ann_path.stem, "error": "no extracted images in folder"})
            continue
        ann_by_frame, counts = read_annotations(ann_path)
        paired, shift, msg = frame_map(images, set(ann_by_frame), offset)
        sequences.append({"name": ann_path.stem, "csv": ann_path, "folder": d,
                          "images": images, "annotations": ann_by_frame,
                          "paired": paired, "offset": shift, "note": msg,
                          "counts": counts, "n_images": len(images),
                          "n_frames_labeled": len(ann_by_frame),
                          "n_paired_frames": len(paired)})
    return sequences


def clip_box(ann: Annotation, width: int, height: int):
    """Keep labeled boxes within the edges of the image."""
    x1 = min(max(ann.x, 0.0), float(width))
    y1 = min(max(ann.y, 0.0), float(height))
    x2 = min(max(ann.x + ann.w, 0.0), float(width))
    y2 = min(max(ann.y + ann.h, 0.0), float(height))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def save_preview(seq, frame_id: int, dst: Path):
    """Draw ground-truth animal and human boxes on one image."""
    img_path = seq["paired"][frame_id]
    with Image.open(img_path) as original:
        canvas = original.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    image_w, image_h = canvas.size
    valid = Counter()
    for ann in seq["annotations"].get(frame_id, []):
        clipped = clip_box(ann, image_w, image_h)
        if clipped is None:
            continue
        cls_name = CLASS_NAMES[ann.cls]
        valid[cls_name] += 1
        draw.rectangle(clipped, outline=PALETTE[ann.cls], width=2)
        label = cls_name.upper()
        tx, ty = int(clipped[0]), int(max(0, clipped[1] - 13))
        draw.text((tx, ty), label, fill=PALETTE[ann.cls], stroke_width=1,
                  stroke_fill="black")
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)
    return valid


def save_previews(sequences, out_dir: Path, n: int):
    """Choose annotated frames with both classes when possible."""
    created = []
    for seq in sequences:
        if "error" in seq or not seq["paired"]:
            continue
        ids = sorted(seq["paired"])
        ids.sort(key=lambda f: (
            -len({a.cls for a in seq["annotations"].get(f, [])}),
            -len(seq["annotations"].get(f, [])), f))
        for frame_id in ids[:min(n - len(created), 1)]:
            dst = out_dir / f"{seq['name']}_frame_{frame_id:06d}.png"
            counts = save_preview(seq, frame_id, dst)
            created.append((dst, counts))
        if len(created) >= n:
            break
    if created:
        thumbs = []
        for path, counts in created:
            with Image.open(path) as im:
                img = im.convert("RGB")
            img.thumbnail((520, 350))
            thumbs.append((path.name, img.copy(), counts))
        tile_w, tile_h, margin = 550, 395, 14
        cols = 2 if len(thumbs) > 1 else 1
        rows = (len(thumbs) + cols - 1) // cols
        montage = Image.new("RGB", (cols * tile_w, rows * tile_h), "#121a28")
        d = ImageDraw.Draw(montage)
        for i, (name, im, counts) in enumerate(thumbs):
            x = (i % cols) * tile_w + margin
            y = (i // cols) * tile_h + margin
            montage.paste(im, (x, y + 24))
            d.text((x, y), name[:70], fill="white")
            d.text((x, y + 30 + im.height),
                   f"Dataset labels: {counts['animal']} animal(s), {counts['human']} human(s)",
                   fill="#c9d5e8")
        montage.save(out_dir.parent / "preview_montage.png")
    return created


def report(sequences, out_dir: Path):
    """Summarize data quality, frame matching, and annotation counts."""
    valid = [s for s in sequences if "error" not in s]
    warnings = [s for s in sequences if "error" in s]
    print(f"\nSequences with images and labels: {len(valid)}")
    print(f"Sequence directories not usable: {len(warnings)}")
    total_img = sum(s["n_images"] for s in valid)
    total_labels = Counter()
    for seq in valid:
        total_labels.update(seq["counts"])
    print(f"Extracted image files: {total_img:,}")
    print(f"Annotation counts (all CSV rows): animals={total_labels['animal']:,} "
          f"humans={total_labels['human']:,}")
    for s in sequences:
        if "error" in s:
            print(f"WARNING: {s['name']}: {s['error']}")
        else:
            print(f"  {s['name']}: {s['n_images']:,} images, "
                  f"{s['n_paired_frames']:,}/{s['n_frames_labeled']:,} labeled "
                  f"frames matched; offset={s['offset']}; {s['note']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    compact = []
    for s in sequences:
        compact.append({k: v for k, v in s.items() if k in
                        ("name", "error", "offset", "note", "n_images", "n_frames_labeled", "n_paired_frames")})
    p = out_dir / "audit.json"
    p.write_text(json.dumps({"source": "BIRDSAI TrainReal only", "sequences": compact,
                             "total_image_files": total_img,
                             "annotation_counts": dict(total_labels)}, indent=2), encoding="utf-8")
    print(f"Audit saved: {p}")
    return valid


def balanced_sample(seq, max_per_video: int, stride: int, rng):
    """Select a manageable collection of frames from one video."""
    ids = sorted(seq["paired"])
    if not ids:
        return []
    # Prefer evenly sampled frames; add frames containing the rarer human class.
    selected = ids[::stride]
    if len(selected) > max_per_video:
        # Evenly spaced across the video (no random burst from one segment).
        selected = [selected[round(i * (len(selected) - 1) / max(1, max_per_video - 1))]
                    for i in range(max_per_video)]
    humans = [f for f in ids if any(a.cls == 1 for a in seq["annotations"].get(f, []))]
    if humans and selected:
        n_extra = min(len(humans), max(1, max_per_video // 4))
        extras = [humans[round(i * (len(humans) - 1) / max(1, n_extra - 1))]
                  for i in range(n_extra)]
        selected = list(dict.fromkeys(extras + selected))[:max_per_video]
    return sorted(set(selected))


def export_yolo(sequences, out_dir: Path, val_fraction: float, seed: int,
                max_per_video: int, stride: int):
    """Write a YOLO-format development dataset split by whole videos."""
    if len(sequences) < 2:
        raise SystemExit("At least two fully matched TrainReal sequences are needed for train/val split.")
    # For credible experiments, abort if substantial data is unmatched rather
    # than silently exporting many image/label pairs with a wrong shift.
    bad = [s for s in sequences if s["n_frames_labeled"] and
           s["n_paired_frames"] / s["n_frames_labeled"] < 0.70]
    if bad:
        print("Cannot safely prepare: these videos matched fewer than 70% of labeled frames:")
        for s in bad:
            print(f"  {s['name']} ({s['n_paired_frames']}/{s['n_frames_labeled']})")
        raise SystemExit("Inspect frame numbering and retry with --frame-offset if needed.")

    rng = random.Random(seed)
    names = sorted(s["name"] for s in sequences if s["paired"])
    if len(names) < 2:
        raise SystemExit("Need >=2 matched videos for a video-level train/val split.")
    rng.shuffle(names)
    val_count = max(1, min(len(names) - 1, round(len(names) * val_fraction)))
    val_names = set(names[:val_count])
    print(f"\nVideo split: {len(names) - val_count} train video(s), {val_count} val video(s)")
    print("Official TestReal is NOT used or copied.")
    target = out_dir / "yolo_real"
    for split in ("train", "val"):
        (target / "images" / split).mkdir(parents=True, exist_ok=True)
        (target / "labels" / split).mkdir(parents=True, exist_ok=True)
    manifest = []
    counters = Counter()
    for s in sequences:
        if not s["paired"]:
            continue
        split = "val" if s["name"] in val_names else "train"
        ids = balanced_sample(s, max_per_video, stride, rng)
        for f in ids:
            image_path = s["paired"][f]
            try:
                with Image.open(image_path) as im:
                    width, height = im.size
            except (OSError, ValueError) as e:
                print(f"WARNING: unreadable {image_path}: {e}")
                counters["unreadable_images"] += 1
                continue
            if width <= 0 or height <= 0:
                continue
            lines = []
            for a in s["annotations"].get(f, []):
                bounds = clip_box(a, width, height)
                if bounds is None:
                    counters["invalid_boxes"] += 1
                    continue
                x1, y1, x2, y2 = bounds
                xc = (x1 + x2) / (2 * width)
                yc = (y1 + y2) / (2 * height)
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                lines.append(f"{a.cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                counters[f"{split}_{CLASS_NAMES[a.cls]}"] += 1
            output_stem = f"{s['name']}__f{f:07d}"
            new_img = target / "images" / split / (output_stem + image_path.suffix.lower())
            new_label = target / "labels" / split / (output_stem + ".txt")
            shutil.copy2(image_path, new_img)
            new_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            manifest.append((split, s["name"], f, image_path.name, len(lines)))
            counters[f"{split}_images"] += 1
        print(f"  exported {split}: {s['name']}, {len(ids)} frames selected")
    yaml_path = target / "data.yaml"
    # Absolute path avoids ambiguous relative-resolution behavior in Ultralytics.
    escaped_path = str(target.resolve()).replace("'", "''")
    yaml_path.write_text(
        f"path: '{escaped_path}'\ntrain: images/train\nval: images/val\n"
        "nc: 2\nnames: [animal, human]\n", encoding="utf-8")
    with (target / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "video", "original_frame_id", "original_image_name", "valid_box_count"])
        w.writerows(manifest)
    (target / "split_videos.json").write_text(json.dumps({
        "seed": seed, "video_level_split": True,
        "train_videos": sorted(set(names) - val_names),
        "validation_videos": sorted(val_names),
        "official_TestReal_used": False,
        "selection": {"max_frames_per_video": max_per_video, "stride": stride}
    }, indent=2), encoding="utf-8")
    print("\nExport complete:")
    print(f"  {yaml_path}")
    for key in sorted(counters):
        print(f"  {key}: {counters[key]:,}")
    print("Use this sample for model DEVELOPMENT, not your final unbiased test score.")


def main():
    """Inspect original BIRDSAI data and optionally export train/val images."""
    p = argparse.ArgumentParser(description="Inspect and prepare BIRDSAI REAL thermal data for ThermalScout-Net")
    p.add_argument("--data", help="BIRDSAI root folder OR TrainReal folder")
    p.add_argument("--mode", choices=["inspect", "prepare"], default="inspect")
    p.add_argument("--out", default="birdsai_step2_output", help="output directory")
    p.add_argument("--frame-offset", type=int, default=None,
                   help="Manual image_file_number - annotation_frame_number (usually 0 or 1)")
    p.add_argument("--previews", type=int, default=4)
    p.add_argument("--max-frames-per-video", type=int, default=120)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not 0.05 <= args.val_fraction <= 0.50 or args.stride < 1 or args.max_frames_per_video < 1:
        p.error("val fraction must be 0.05-0.50, stride >=1, max frames per video >=1")
    train = locate_data(args.data)
    print(f"Using real training partition: {train}")
    print("Classes from official BIRDSAI format: animal=0, human=1")
    print("Dataset originals are READ ONLY; TestReal is held out.")
    sequences = collect_sequences(train, args.frame_offset)
    out = Path(args.out).expanduser().resolve()
    valid = report(sequences, out)
    previews = save_previews(valid, out / "previews", args.previews)
    for path, classes in previews:
        print(f"Preview: {path}  labels={dict(classes)}")
    if not previews:
        print("No annotated images could be previewed. Check the audit above.")
    if args.mode == "prepare":
        broken = [s for s in sequences if "error" in s]
        if broken:
            raise SystemExit("Preparation stopped: missing image folders or unreadable videos are listed "
                             "above. Resolve these first so no videos are silently omitted.")
        export_yolo(valid, out, args.val_fraction, args.seed,
                    args.max_frames_per_video, args.stride)
    else:
        print("\nNEXT: Open the saved preview images and make sure every box is on the correct object.")
        print("Once validated, run again with --mode prepare to export the training/validation subset.")


if __name__ == "__main__":
    main()
