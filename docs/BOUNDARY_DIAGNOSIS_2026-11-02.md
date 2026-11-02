# Human and boundary diagnosis — November 2, 2026

Inspected frozen baseline and focal prediction caches on the same 334 development validation images. No retraining or defaults changed; official TestReal was not accessed. Names and letterbox-adjusted labels were checked against the current dataset, all input bytes fingerprinted, and baseline aggregate precision/recall plus class TP/FP/GT reproduced against saved evaluations.

## Controlled boundary ablation

At confidence .25 and IoU .50, compare the original retained predictions with two fixed rules: remove centers outside the exact letterboxed content rectangle, or remove centers outside that rectangle inset by four pixels on all sides at input size 256. The latter excludes real image content, including legitimate edge objects; it is a diagnostic ablation, not a recommended default. Four pixels was fixed before measuring this band, with no margin sweep.

| Model / rule | TP | FP | Precision | Recall | Human TP / 62 |
|---|---:|---:|---:|---:|---:|
| Focal / original | 441 | 12,919 | 3.30% | 43.41% | 4 |
| Focal / outside padding only | 441 | 12,911 | 3.30% | 43.41% | 4 |
| Focal / four-pixel boundary band | 439 | 6,193 | 6.62% | 43.21% | 4 |
| Baseline / original | 438 | 12,880 | 3.29% | 43.11% | 12 |
| Baseline / outside padding only | 438 | 12,878 | 3.29% | 43.11% | 12 |
| Baseline / four-pixel boundary band | 434 | 11,624 | 3.60% | 42.72% | 12 |

The focal boundary band removes 6,726 FP (52.06%) and two TP. Baseline loses 1,256 FP and four TP. Only eight focal FP centers are outside content: the visible boundary artifact is mainly just inside image content, not in the padding itself. All 1,016 label centers are inside content. This check does not establish full annotation alignment/coverage. After filtering, many false positives remain. The difference between models supports investigating focal loss and boundary features but does not establish causation. Filtering after the existing 500-candidate and 40-output caps cannot recover excluded detections; AP and speed were not measured.

## Human overlap diagnosis

For each human label, inspect retained predictions at confidence .25 and overlap IoU .50, independent of one-to-one matching:

| Model | Human-class overlap | Animal-only overlap | Neither class at .50 | Any-class overlap at .10 |
|---|---:|---:|---:|---:|
| Focal | 4 | 17 | 41 | 59 |
| Baseline | 12 | 10 | 40 | 62 |

Animal-only means at least one overlapping animal prediction and no overlapping human prediction. These are label-wise diagnostic counts, not mutually exclusive prediction assignments or proof that reclassifying animal boxes would recover every label. The .10 column overlaps the other categories. Both classification and localization/missing-candidate behavior matter. The caches cannot reveal the human alternative class score discarded by argmax or detections dropped upstream. Boundary removal does not recover any additional humans.

The two example frames are adjacent and intentionally human-heavy; they are not independent representative samples. In the new comparison, columns are labels, original predictions at .25, outside-padding filter at .25, and four-pixel boundary filter at .25. Blue is human and green is animal. Images remain Git-ignored under `training_runs/20261102_focal_boundary_final/human_padding_comparison.png`.

## Interpretation and next training experiment

Retain the original baseline as the human-detection reference. Do not replace defaults with the boundary crop or declare the focal model superior. A useful next controlled training experiment would vary class loss weight alone against baseline objectness, holding seed, data, image size, optimizer and budget fixed. Inspect human/animal box sizes and annotation mapping first; the existing human weight is already 2. Treat background rejection as a separate subsequent change. Compare both class AP and precision/recall, and report edge-object losses. This diagnostic experiment did not train a new model.

## Reproduction and evidence

Run from the repository root. Use a fresh output folder each time:

```powershell
py -3.12 diagnose_padding.py --cache records/experiments/20261102_focal_threshold_sweep/predictions.json --checkpoint training_runs/20261029_focal_experiment/thermal_scout_real_weights.pt --out training_runs/my_boundary_check
py -3.12 -m pytest tests -q
py -3.12 project_manager.py update
ii .\training_runs\my_boundary_check\human_padding_comparison.png
```

For baseline, use cache `training_runs/20261029_133235_full/error_analysis/predictions.json` and checkpoint `training_runs/20261029_133235_full/thermal_scout_real_weights.pt`. If a cache is missing, regenerate it with `analyze_validation.py` for that checkpoint first. This analysis uses the existing frozen caches; it does not independently regenerate model outputs or prove a cache/checkpoint association beyond reproducing saved metrics and recording their hashes.

[Focal evidence](../records/experiments/20261102_boundary_diagnosis/focal_summary.json) and [baseline evidence](../records/experiments/20261102_boundary_diagnosis/baseline_summary.json) include exact counts and checkpoint, source, cache and current dataset hashes. Ten tests passed, including content-rectangle boundaries and human-versus-animal overlap categories. Analysis reruns reproduced the same counts. Report section 12 persists on rebuild.
