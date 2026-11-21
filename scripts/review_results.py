"""Read saved development results. Optionally audit an existing BIRDSAI prediction cache."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "records" / "experiments"
REPORTS = {
    "ThermalScout-Net": ROOT / "20261112_resolution512_human4/evaluation.json",
    "YOLO reference": ROOT / "20261112_reference_yolo26n/reference_evaluation.json",
}


def verify_checkpoint(path):
    """Check the file against the recorded run without loading or changing it."""
    report = json.loads(REPORTS["ThermalScout-Net"].read_text())
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != report["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint differs from the recorded run: SHA-256 {digest}")
    print(f"Verified epoch {report['checkpoint_epoch']} checkpoint: {digest}")


def show_summary():
    print("Saved development results | confidence 0.25 | IoU 0.50")
    for model, path in REPORTS.items():
        report = json.loads(path.read_text())
        for label, values in report["class_metrics"].items():
            tp, fp, gt = values["tp"], values["fp"], values["gt"]
            precision = f"{tp / (tp + fp):.1%}" if tp + fp else "n/a"
            recall = f"{tp / gt:.1%}" if gt else "n/a"
            print(f"{model:16} {label:6} {values['tp']}/{values['gt']} found, "
                  f"{values['fp'] / report['evaluation_images']:.2f} false boxes/frame, "
                  f"precision {precision}, recall {recall}")


def audit_cache(entries):
    from thermalscout.evaluate_detector import get_counts

    videos = defaultdict(list)
    seen = set()
    for entry in entries:
        name = entry["image"]
        video, separator, frame = Path(name).stem.rpartition("__f")
        if not video or not separator or not frame.isdigit() or name in seen:
            raise ValueError(f"Expected a unique BIRDSAI video__fNUMBER image: {name}")
        seen.add(name)
        videos[video].append(entry)

    if not videos:
        raise ValueError("Empty prediction cache")
    for video, frames in sorted(videos.items()):
        for c, label in enumerate(("animal", "human")):
            counts = get_counts(frames, c, conf=0.25)
            print(f"{video} {label}: {len(frames)} frames, {counts['tp']}/{counts['gt']} "
                  f"found, {counts['fp'] / len(frames):.2f} false boxes/frame")

    # Score matching is greedy in descending confidence. Counts above successive
    # cutoffs can therefore be subtracted without matching one label twice.
    for c, label in enumerate(("animal", "human")):
        scores = [score for entry in entries for cls, score, _ in entry["predictions"] if cls == c]
        if any(not 0 <= score <= 1 for score in scores):
            raise ValueError("Scores must be finite and between zero and one")
        previous_hits = 0
        for bucket in reversed(range(10)):
            low = bucket / 10
            high = (bucket + 1) / 10
            observed = [s for s in scores if low <= s and (s < high or bucket == 9)]
            hits = get_counts(entries, c, conf=low)["tp"]
            if observed:
                print(f"{label} bin {bucket}/10: n={len(observed)}, "
                      f"mean score={sum(observed)/len(observed):.3f}, "
                      f"fraction matched={(hits-previous_hits)/len(observed):.3f}")
            previous_hits = hits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, help="predictions.json from analyze_validation")
    parser.add_argument("--checkpoint", type=Path, help="Verify a copy of the recorded 512-pixel checkpoint")
    args = parser.parse_args()
    if args.checkpoint:
        try:
            verify_checkpoint(args.checkpoint)
        except (OSError, ValueError) as error:
            parser.exit(1, f"{error}\n")
    show_summary()
    if args.cache:
        audit_cache(json.loads(args.cache.read_text()))
