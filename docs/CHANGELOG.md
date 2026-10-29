# Project development history

## 2026-10-28 — Verified earlier progress
- Custom synthetic CNN (29,635 trainable parameters) trained for eight epochs.
- Eight recorded training losses: 0.9574, 0.3722, 0.2400, 0.1815, 0.1320, 0.0702, 0.0278, 0.0220.
- BIRDSAI TrainReal inspection: 32 sequences, 40,661 image frames, 74,668 animal annotations and 12,531 human annotations.
- Video-based development export: 2,001 training images and 334 validation images.
- Official TestReal footage not used.

## 2026-10-28 — Repository organization
- Separated real-detector Python code into model, data, training, inference, and evaluation modules.
- Added simple function explanations, repository README, data safety exclusions, and GitHub instructions.
- Kept original synthetic demo and prior reports as references.
- Improved detection candidate selection to prioritize confidence rather than image location.
- **No real-image model accuracy has been measured or claimed.**

## 2026-10-29 — Completed real-image development baseline
- Completed run 20261029_133235_full: 10 epochs, 2,001 training images, 334 validation images; best checkpoint at epoch 7.
- Measured precision 3.29%, recall 43.11%, mAP@0.50 8.11%, mAP@0.50:0.95 2.44%, and CPU inference plus postprocessing at 7.11 FPS.
- Added portable evaluation JSON and full epoch history under records/experiments, plus a results summary in docs/RESULTS_2026-10-29.md.
- Results use development validation only. Official TestReal remains held out; high false-positive counts require further work.
