"""Optional pretrained-detector comparison on the same BIRDSAI development split.

Training and validation run locally. This never reads official TestReal.
Install requirements-reference.txt separately from the custom CNN requirements.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from thermalscout.data import BirdsaiDataset
from thermalscout.evaluate_detector import ap_101, get_counts


def load_yolo():
    try:
        from ultralytics import YOLO, __version__
    except ImportError as error:
        raise SystemExit('Install optional requirements-reference.txt first') from error
    return YOLO, __version__


def evaluate(weights, data, output, device):
    YOLO, version = load_yolo()
    model = YOLO(str(weights))
    dataset = BirdsaiDataset(data, 'val', 512, 0, 43)
    entries = []
    started = time.perf_counter()
    for index, (path, _, boxes) in enumerate(dataset.records, 1):
        result = model.predict(str(path), imgsz=512, conf=0.001, iou=0.5, max_det=40,
                               device=device, verbose=False)[0]
        width, height = result.orig_shape[1], result.orig_shape[0]
        detections = []
        for cls, score, xyxy in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                                     result.boxes.xyxy.tolist()):
            if int(cls) in (0, 1):
                x1, y1, x2, y2 = xyxy
                detections.append((int(cls), float(score),
                                   [x1/width, y1/height, x2/width, y2/height]))
        entries.append({'ground_truth': boxes, 'predictions': detections})
        if index % 50 == 0:
            print(f'Evaluated {index}/{len(dataset)} original validation frames', flush=True)
    elapsed = time.perf_counter() - started
    scores = {}
    for cls, name in enumerate(('animal', 'human')):
        row = get_counts(entries, cls)
        row['ap50'] = ap_101(entries, cls, 0.5)
        row['map5095'] = float(np.mean([ap_101(entries, cls, threshold)
                                        for threshold in np.arange(0.5, 1.0, 0.05)]))
        scores[name] = row
    result = {'model': 'YOLO26n reference', 'ultralytics_version': version,
              'split': 'BIRDSAI video-held-out development validation',
              'evaluation_images': len(dataset), 'image_size': 512,
              'confidence_for_counts': 0.25, 'iou_for_counts': 0.5,
              'map50': float(np.mean([row['ap50'] for row in scores.values()])),
              'class_metrics': scores, 'fps_end_to_end': len(dataset)/elapsed,
              'fps_scope': 'image loading, model prediction and postprocessing'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"mAP50={result['map50']:.4f}; human TP={scores['human']['tp']}; "
          f"human FP={scores['human']['fp']}; saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/processed/yolo_real'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--weights', type=Path, help='Evaluate an existing YOLO checkpoint')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    root = args.data.resolve()
    if not (root / 'split_videos.json').is_file() or not (root / 'data.yaml').is_file():
        parser.error('Expected the prepared BIRDSAI development dataset')
    if args.weights:
        weights = args.weights.resolve()
    else:
        if args.epochs < 1 or args.out.exists():
            parser.error('Choose a fresh --out directory and positive --epochs')
        args.out.mkdir(parents=True)
        local_yaml = args.out / 'dataset_current_path.yaml'
        dataset_path = str(root).replace("'", "''")
        local_yaml.write_text(f"path: '{dataset_path}'\ntrain: images/train\nval: images/val\n"
                              "nc: 2\nnames: [animal, human]\n", encoding='utf-8')
        YOLO, _ = load_yolo()
        model = YOLO('yolo26n.pt')
        model.train(data=str(local_yaml), imgsz=512, epochs=args.epochs,
                    batch=4, seed=42, patience=5, device=args.device,
                    project=str(args.out.resolve().parent), name=args.out.name,
                    exist_ok=True, pretrained=True, verbose=False)
        weights = args.out.resolve() / 'weights' / 'best.pt'
    if not weights.is_file():
        raise SystemExit(f'Checkpoint missing: {weights}')
    evaluate(weights, root, args.out.resolve() / 'reference_evaluation.json', args.device)


if __name__ == '__main__':
    main()
