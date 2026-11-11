# Focal checkpoint threshold experiment — November 2, 2026

Frozen checkpoint: `20261029_focal_experiment`, epoch 7. All 334 development validation frames were evaluated on CPU without retraining. Official TestReal was not used. Training and inference defaults remain unchanged.

| Operating rule (animal / human confidence) | Precision | Recall | F1 | TP | FP | Human TP / labels |
|---|---:|---:|---:|---:|---:|---:|
| Baseline .25 / .25 | 3.30% | 43.41% | 6.14% | 441 | 12,919 | 4 / 62 |
| Global .99 / .99 | 32.44% | 25.10% | 28.30% | 255 | 531 | 0 / 62 |
| Human-preserving .99 / .25 | 28.52% | 25.49% | 26.92% | 259 | 649 | 4 / 62 |

The class-specific rule removes 94.98% of baseline false positives but loses 182 true positives. It preserves only the focal checkpoint's already-low 6.45% human recall; it does not restore the baseline-trained model's 19.35% human recall. These are validation-selected operating points, not model improvements or unbiased test results. Do not install either rule as a new default from this experiment alone.

**Control and selection.** Generated predictions once at confidence .001, then filtered the retained boxes at 20 thresholds (.001 through 1.0). IoU .50, top 500 candidates, classwise NMS at .50, and 40 detections/image were fixed. Baseline aggregate precision/recall and per-class TP/FP/GT exactly reproduced the original evaluation. The global rule maximizes validation micro F1 over the tested grid. The guarded rule maximizes that score with human confidence fixed at .25. Animal prevalence dominates micro F1. AP/mAP was not recalculated as an operating-point improvement: filtering ranked outputs does not demonstrate improved learned ranking. No speed comparison was made.

**Checkpoint audit.** The full folders `20261029_125035_full` and `20261029_133235_full` contain byte-identical checkpoint files and identical model tensors. The earlier folder lacks a loss-history CSV, so they cannot establish two independent completed training experiments. All three quick folders (`20261029_124654_quick`, `20261029_124946_quick`, `20261102_114742_quick`) have identical model tensors and loss histories. The two October files are byte-identical; November has different serialization metadata but the same learned state. Fixed seed 42 and identical quick settings can reproduce these weights; hashes alone cannot prove copying versus rerunning. The focal weights differ from all baseline and quick weights. Different FPS values do not establish different models.

The training log also contains repeated entries for the same run/history when the drive changed from E: to D:. Historical records were preserved. Registration now recognizes a matching run ID and history hash across relocated paths, preventing additional relocation duplicates. Existing totals are counts of log entries, not unique trained models. Original metrics use generic full/quick names; identify the experiment using its evidence folder.

**Diagnosis.** Of 12,919 focal false positives, overlap heuristics classify 11,627 as background-like, 1,175 localization-like, 92 duplicates and 25 wrong-class. These categories describe overlap with annotations, not visually confirmed causes. Duplicate suppression is not the dominant measured error. Code inspection shows background objectness loss is averaged separately and multiplied by .25; focal loss further downweights easy negatives. Poor background rejection/calibration is a plausible explanation, not established causation. Many false positives survive high confidence.

Training has 6,691 animal versus 1,090 human labels; validation has 954 versus 62. Class loss already weights humans by 2, so missing weighting alone is not the diagnosis. One box per stride-4 cell, first-label wins on collisions, small resized targets, argmax-only class predictions, and the shared 500-candidate/40-output caps can constrain human detection. The sweep cannot recover human boxes excluded upstream. Inspect label alignment, annotation coverage, object sizes and cap occupancy before changing training. The focal run's 4/62 human detections versus baseline 12/62 argues against selecting it solely for aggregate mAP. Per-class results and repeated controlled training seeds are needed for the next comparison.

**Evidence.** [Summary and provenance](../records/experiments/20261102_focal_threshold_sweep/summary.json), [threshold metrics](../records/experiments/20261102_focal_threshold_sweep/thresholds.csv), [class-specific policies](../records/experiments/20261102_focal_threshold_sweep/class_policies.csv), [checkpoint fingerprints](../records/experiments/20261102_focal_threshold_sweep/checkpoint_audit.json), and [validation input hashes](../records/experiments/20261102_focal_threshold_sweep/dataset_manifest.json). The summary records Python/library versions, source hashes, checkpoint hash and prediction hash. Cached predictions and BIRDSAI diagnostic images remain local and Git-ignored.

**Reproduce from the repository root.** Use Python 3.12 with the project's installed dependencies. Choose a fresh output directory; explicit `--out` refuses overwriting an existing directory.

```powershell
py -3.12 -u -m scripts.analyze_validation --run training_runs/20261029_focal_experiment --data data/processed/yolo_real --out training_runs/threshold_repeat_01
py -3.12 -m pytest tests -q
py -3.12 project_manager.py update
ii .\report\Living_Project_Report.html
```

The analysis writes `thresholds.csv`, `class_policies.csv`, `summary.json`, `checkpoint_audit.json`, `dataset_manifest.json`, the prediction cache, tradeoff plot and diagnostic images. Compare prediction hashes and counts against the saved evidence. Keep the checkpoint/data/source versions fixed. The living report includes this document each time it rebuilds, so the findings persist. Earlier analysis behavior without `--out` remains available. No retraining is needed for this experiment.

**Validation completed:** Two full inference passes produced identical prediction hashes and selected metrics. Independently applying the class thresholds reproduced 259 TP / 649 FP and 4 human TP. All eight tests passed, including relocated-history registration and class filtering. Both Markdown and HTML reports were rebuilt and checked for the experiment table.
