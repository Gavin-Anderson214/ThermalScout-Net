# Desktop project verification — November 18, 2026

The December 2 course submission concerns a custom thermal object detector trained and evaluated on the desktop. Raspberry Pi deployment, thermal-camera integration, drone mounting and flight testing are future work.

## What was preserved

The original seven-convolution CNN has 88,503 trainable parameters and classes `0 = animal`, `1 = human`. BIRDSAI remains the primary dataset. BAMBI, other thermal data experiments and the pretrained reference remain separate studies. Historical datasets, checkpoints and experiment records were preserved.

## Dataset verification

The local original **TrainReal** extraction contains 40,661 images across 32 sequences. Its source CSVs contain 74,668 usable animal and 12,531 usable human annotations, without exact duplicate annotation rows. These are counts of this extracted training partition, not the full published BIRDSAI dataset.

| Prepared split | Images / label files | Animal boxes | Human boxes | Video IDs |
|---|---:|---:|---:|---:|
| Training | 2,001 / 2,001 | 6,691 | 1,090 | 26 |
| Development validation | 334 / 334 | 954 | 62 | 6 |

All 2,335 prepared image files match the corresponding source images byte for byte. Every prepared label matches its original CSV box after clipping, normalization and six-decimal rounding. The audit found zero duplicate annotation rows, invalid/out-of-bounds boxes, missing manifest records, orphan labels, manifest count mismatches, duplicate image bytes or duplicate decoded grayscale images. No boxes collide in the stride-4 target grid at 512 pixels. There are no entirely empty frames in the prepared split.

The source uses pixel `min_x, min_y, width, height`. Prepared labels use `class, normalized_center_x, normalized_center_y, normalized_width, normalized_height`. Both source and model use animal first, human second. [Official class mapping](https://lila.science/datasets/conservationdrones) and [source coordinate/indexing documentation](https://github.com/exb7900/BIRDSAI) agree with the converter.

Training and validation **video IDs are disjoint**. However, IDs beginning with `0000000067` appear in both splits. The manifest provides no flight identity, so flight-level independence is unproven. All 62 human validation boxes come from video `0000000067_0000000059`. Scores describe this repeatedly used development split and cannot establish general performance across independent flights. Official TestReal was not read or used in this work.

The local `data.yaml` had an obsolete E: drive path. The custom loader uses the explicitly supplied dataset root rather than that YAML path, so it did not redirect these experiments. The path now points to the current D: location. The original metadata is captured in the audit evidence.

## Training and inference contract

- A 256-pixel square produces a `7 × 64 × 64` output. A 512-pixel square produces `7 × 128 × 128`.
- Both paths convert to grayscale, preserve aspect ratio, resize with bilinear interpolation, add centered black padding and divide pixels by 255. The audit compared both paths on every prepared frame without a pixel mismatch.
- A labeled box belongs to its center cell. Targets contain fractional cell offsets and log width/height in grid-cell units. Inference applies sigmoid to offsets and exponential to width/height, normalizes the box and clips its corners.
- Confidence is sigmoid objectness multiplied by the largest softmax class probability. It is a ranking score without probability calibration.
- NMS uses IoU 0.50 separately within each class. Inference considers the top 500 candidates and retains at most 40 detections per frame. AP evaluation starts at score 0.001. Precision, recall and error counts use score 0.25 and IoU 0.50.
- Training calls `model.train()`. Validation and inference call `model.eval()` without gradient updates. Regression tests check that validation preserves weights and batch-normalization statistics.
- Checkpoint loading now checks class order, image size and strict weight compatibility. Single-image inference explicitly rejects tiled checkpoints to prevent silently applying the wrong preprocessing path.

## Real-image memorization gate

Both tests used the same 16 BIRDSAI **training** images: eight with human labels and eight animal-only frames, selected evenly from their respective lists. They contain 29 animal and 15 human boxes. Settings were 512 pixels, seed 42, batch 4, AdamW, learning rate 0.001, weight decay 0.0001, human class weight 4, background weight 0.25, no augmentation and at most 400 epochs. These tests measure memorization, not generalization.

The original baseline loss reached 100% ranked-box mAP@50 but retained false positives at confidence 0.25. It failed the stricter gate requiring mAP50 ≥95%, each class recall ≥95%, each class precision ≥90%, and non-identical confidence scores. Low training loss alone therefore did not establish acceptable detections.

Changing **only objectness background treatment** to the existing hard-negative option passed at epoch 80:

| Memorization measurement | Animal | Human |
|---|---:|---:|
| Correct / labeled boxes | 29 / 29 | 15 / 15 |
| False positives at confidence 0.25 | 0 | 0 |
| Precision / recall | 100% / 100% | 100% / 100% |
| AP50 | 100% | 100% |

Evaluation loss fell from 2.9450 to 0.0845. The 44 scores range from 0.927496 to 0.996437 and are all distinct. The model learns boxes and both classes without an architecture change. Hard-negative mining concentrates background loss on the strongest errors, selecting at least 32 cells or three negatives per positive in each batch. This addresses dilution of confident false alarms when averaging over every background cell.

The baseline diagnostic's original saved checkpoint selected the first maximum mAP tie, so its summary can retain more false positives than its last epoch. Its full history is preserved. The helper now breaks mAP ties by minimum per-class precision.

## Controlled full-data experiments

New full-data runs hold architecture, split, seed 42, 512-pixel input, batch 4, human weight 4, background weight 0.25, optimizer, learning rate, horizontal-flip augmentation and 10-epoch budget fixed. The only training-variable change is baseline versus hard-negative objectness loss. Every epoch records training loss, validation loss, per-class precision/recall, AP50, true positives and false positives. Misses equal labeled boxes minus true positives. The new `--select-by map50` option selects by measured detection mAP rather than validation loss.

| Candidate | Selected epoch | mAP50 | Animal AP50 | Human AP50 | Animal TP / FP | Human TP / FP |
|---|---:|---:|---:|---:|---:|---:|
| Historical 512px | 4 | **29.11%** | 22.87% | 35.35% | 683 / 12,561 | 31 / 53 |
| New baseline | 5 | 21.69% | 40.70% | 2.68% | 723 / 12,471 | 14 / 131 |
| New hard-negative | 10 | 22.98% | 45.95% | 0.00% | 688 / 12,671 | 0 / 1 |

All three saved checkpoints were independently reevaluated on the same 334 frames on November 18. The two new runs each completed ten epochs. The historical run used minimum validation-loss selection and stopped after nine epochs, so it is a historical reference rather than part of the isolated new loss ablation. Changing checkpoint selection and GPU numerical behavior also mean the new baseline is not an exact reproduction of the historical result. The two new runs used four CPU threads; GPU kernels are not forced deterministic. A single seed cannot estimate result variance.

The **historical epoch-4 checkpoint remains selected** by the highest reevaluated mAP50. The new hard-negative run improves animal AP relative to the new baseline but loses human detection. Passing memorization does not establish validation performance, so the hard-negative option was not made the training default or used to replace the final model.

### Selected model at confidence 0.25

| Class | Correct | False positives | Missed | Precision | Recall | AP50 |
|---|---:|---:|---:|---:|---:|---:|
| Animal | 683 | 12,561 | 271 | 5.16% | 71.59% | 22.87% |
| Human | 31 | 53 | 31 | 36.90% | 50.00% | 35.35% |

Overall mAP50 is **29.1063%**, mAP50:95 is **6.3458%**, micro precision is **5.3571%**, and micro recall is **70.2756%**. Measured forward-pass plus postprocessing throughput was 21.64 frames/second on this desktop, excluding file I/O and dataset loading. It is not end-to-end camera throughput. The checkpoint SHA-256 is `863a403942fe8345c3ae486e399c88b705a38fe019b9d9ab131ec54f517c4d57`.

An exploratory threshold sweep on the same development predictions found animal F1 highest among the tested cutoffs at 0.995 (499 TP, 1,250 FP, 455 missed, 28.53% precision, 52.31% recall) and human F1 highest at 0.65 (30 TP, 23 FP, 32 missed, 56.60% precision, 48.39% recall). These cutoffs were selected after seeing validation results. They are optimistic development operating points, not independently validated settings. Even after thresholding, false alarms remain substantial. The headline results and visual examples retain the fixed 0.25 cutoff.

**Reliability remains unfinished.** The detector is a measured, functioning course prototype and meets the software/memorization and both-class-detection checks. It does not yet meet a reasonable practical reliability standard. The remaining work requires better background discrimination and evaluation on footage with verified flight identities and human labels across more independent scenes. Additional epochs alone are not an established solution.

### Evidence and final files

- [Audit](../records/experiments/20261118_final/audit.json), [memorization result](../records/experiments/20261118_final/overfit_hardnegative_summary.json), and [full checkpoint comparison](../records/experiments/20261118_final/selection.json).
- [Baseline epoch measurements](../records/experiments/20261118_final/baseline_detection_history.json) and [hard-negative epoch measurements](../records/experiments/20261118_final/hardnegative_detection_history.json).
- [Threshold measurements](../records/experiments/20261118_final/thresholds.csv) and [example selection, boxes and scores](../records/experiments/20261118_final/example_manifest.json).
- Selected weights: `training_runs/20261118_final/thermal_scout_real_weights.pt` (local and ignored).
- Complete local prediction cache: `training_runs/20261118_final/predictions.json`.
- Fully labeled local examples and presentation overviews: `training_runs/20261118_final_examples_v4/`.
- [Final presentation](../report/presentations/ThermalScout-Net_Desktop_Final_2026-11-18.pptx). The source PowerPoint was preserved.

The animal and human success examples maximize matched boxes for their respective class. The false-positive example maximizes the number of false boxes, and the missed-human example maximizes missed human boxes. Ties use the first frame in dataset order. Every box appears in the overview; numbered highlights link to readable class/score captions on the slides. These are illustrative selected cases, not an estimate of average quality.

For a desktop demonstration of the selected checkpoint:

```powershell
python -m thermalscout.predict_image --checkpoint training_runs/20261118_final/thermal_scout_real_weights.pt --image data/processed/yolo_real/images/val/0000000067_0000000059__f0000636.jpg --out training_runs/NEW_FINAL_DEMO --threshold 0.25 --device cuda
```

Use a new output directory each time. The saved preview contains predictions, not verified labels. Use the evaluator and example script for truth-matched TP/FP/FN pictures.

## Submission status and remaining work

Completed: source/prepared data audit, format and model-mode checks, real-image memorization, two bounded GPU experiments, detection-based checkpoint selection, final metrics, both-class validation detections, annotated success/error examples, documentation and presentation.

Before December 2: rehearse the slides and demo, review false alarms and misses, and describe the reliability limitation explicitly. Any further accuracy claim needs a controlled experiment and measured detection results. Verified flight independence and practical reliability are unresolved, even though the course submission now contains reproducible evidence.

After the desktop submission: Raspberry Pi performance, thermal-camera integration, drone mounting and flight tests. No code or hardware work for those tasks was required in this session.

## Reproduce the checks

Run from the repository root with CUDA-enabled PyTorch. Choose fresh output directories.

```powershell
python -m pytest
python -m scripts.verify_pipeline audit --source "PATH_TO_BIRDSAI/TrainReal" --out training_runs/NEW_AUDIT
python -m scripts.verify_pipeline overfit --objectness hard-negative --out training_runs/NEW_OVERFIT
python -m thermalscout.train_detector --full --size 512 --batch 4 --epochs 10 --human-weight 4 --seed 42 --device cuda --select-by map50 --objectness hard-negative --out training_runs/NEW_RUN
python -m thermalscout.evaluate_detector --checkpoint training_runs/NEW_RUN/thermal_scout_real_weights.pt --data data/processed/yolo_real --out training_runs/NEW_RUN/evaluation.json --predictions training_runs/NEW_RUN/predictions.json --device cuda
python -m scripts.final_examples --predictions training_runs/NEW_RUN/predictions.json --out training_runs/NEW_RUN/examples --size 512
```

Model checkpoints, prediction caches and dataset-derived example images stay local under ignored `training_runs/`. Compact measurements and configurations belong in `records/experiments/20261118_final/`. The final presentation includes the measured examples and explanations on its slides.
