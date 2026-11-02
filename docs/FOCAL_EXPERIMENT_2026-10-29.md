# Focal negative-objectness experiment — October 29, 2026

A controlled development experiment tested whether weighting difficult negative cells reduces false positives. The focal option multiplies each negative object's binary-cross-entropy loss by predicted objectness squared; the existing 0.25 negative-loss multiplier is retained. Baseline remains the default.

Both runs used the same 2,001 training and 334 validation images, video-separated split, CPU, seed 42, input size 256, batch size 4, learning rate, architecture, random initialization, and 10-epoch budget. Each run's checkpoint was selected using its own validation loss (epoch 7 in both runs). Because the loss function differs, those loss values are not directly comparable. Neither experiment used official TestReal.

| Development metric | Baseline | Focal negative loss |
|---|---:|---:|
| Micro precision (confidence .25 / IoU .50) | 3.29% | 3.30% |
| Micro recall | 43.11% | 43.41% |
| mAP@.50 | 8.11% | 8.99% |
| mAP@.50:.95 | 2.44% | 2.09% |
| Animal precision / recall | 3.25% / 44.65% | 3.30% / 45.81% |
| Animal AP@.50 | 14.07% | 17.73% |
| Human precision / recall | 6.03% / 19.35% | 3.25% / 6.45% |
| Human AP@.50 | 2.15% | 0.25% |
| Human true positives / labels | 12 / 62 | 4 / 62 |
| All false positives | 12,880 | 12,919 |

The focal variant improved animal AP@.50 but worsened human detection and did not reduce total false positives. This single-seed experiment is not evidence that the focal variant is generally better; retain the baseline. The human sample is small and metrics have uncertainty. Scores are development validation, not final test results.

Evidence: [focal evaluation](../records/experiments/20261029_focal_experiment/evaluation.json), [epoch history](../records/experiments/20261029_focal_experiment/loss_history.csv), [baseline evaluation](../records/experiments/20261029_133235_full/evaluation.json). Weights and dataset remain in their local Git-ignored directories. The evaluation JSON's generic model label is supplied by the existing evaluator; this document and the folder name identify the focal run.

To reproduce from the project root:

```powershell
py -3.12 -u -m thermalscout.train_detector --full --data data/processed/yolo_real --out training_runs/focal_new --objectness focal --device cpu
py -3.12 -u -m thermalscout.evaluate_detector --checkpoint training_runs/focal_new/thermal_scout_real_weights.pt --data data/processed/yolo_real --out training_runs/focal_new/evaluation.json --device cpu
```

Six core tests passed, including a check that a high-confidence background mistake receives a much larger gradient than an easy negative. The next research direction is to inspect annotation coverage and border/padding errors and consider class-specific handling; evaluate any change against both classes, rather than selecting solely by overall mAP.
