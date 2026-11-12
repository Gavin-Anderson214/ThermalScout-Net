# Improving small-object predictions

The canonical BIRDSAI development split has 2,001 train and 334 validation frames. All 62 validation human boxes come from one video. Official TestReal is still untouched. At 256-pixel input, the median human short side is 4.4 pixels in training and 7.6 pixels in validation; the detector predicts at stride 4. The original selected frames contain no entirely empty images. These facts motivate testing preserved pixel detail and verified negative scenes.

## Stopped continuous run and measured baseline

`training_runs/20261112_025045_full` stopped after 97 completed CUDA epochs. Its best validation-loss checkpoint is epoch 5 (0.7608), while later training loss kept falling and validation loss worsened. The CSV and best weights are local under that run. Full-frame evaluation of the saved checkpoint on all 334 development frames gave 7.15% mAP@50, 7/62 human true positives, 75 human false positives, and 12,645 animal false positives at confidence .25. These are frame-level counts, not unique individuals. The run used human weight 2 and is not directly interchangeable with the earlier weight-4 baseline (15.54% mAP@50).

## New controlled options

- `--patience N` ends a run after N epochs without lower validation loss. Best weights and epoch history are saved. A bounded `--epochs` run is easier to compare than continuous training.
- `--tile` trains on overlapping 256-pixel native-resolution crops when `--size 256` is used. Empty crops from labeled training images are retained. The saved checkpoint records this choice; the evaluator then tiles each original validation frame, merges detections, and scores original full-frame boxes. Tiled evaluation takes multiple model passes per frame and must be benchmarked for speed.
- `--verified-backgrounds DIR` accepts a directory of *manually checked* empty images from training videos. The loader adds them only to training. Missing labels are never silently treated as proof that a scene is empty. Keep validation and official test footage out of that directory.
- `scripts.rank_training_frames` scores prepared training frames for missed humans/animals and false boxes, then limits how many nearby frames from one video appear in its review queue. Inspect both image and original labels before correcting or adding data. This is a review aid, not an automatic labeling system.

## First measured size experiment

A full-data 512-pixel run with human weight 4, seed 42 and CUDA stopped after epoch 9 by patience 5. Its best checkpoint came from epoch 4. On the same 334 development frames, it measured 29.11% mAP@50, with 31/62 human true positives, 53 human false positives, 683/954 animal true positives and 12,561 animal false positives at confidence .25. Human AP50 was 35.35%; animal AP50 was 22.87%. Compared with the historical 256-pixel, weight-4 run (15.54% overall mAP@50, 21 human TP and 492 human FP), the size change is promising for human detection in this single-seed development comparison. Animal AP50 did not improve, and animal false alarms remain a major issue. Throughput measurements include postprocessing and cannot isolate the effect of image size.

The tiled path passed a 160-source-image training smoke check and 10-original-frame evaluation smoke check. These smoke scores are not comparable to full-data results. That check measured roughly 2 original frames/second on the GPU, so full tiled training is deferred until a quality-versus-speed comparison is warranted. The 512-pixel full-image checkpoint is the current candidate for that comparison.

## Recommended experiment sequence

Run these from the repository root, with a fresh output directory each time. Use a 10-epoch budget, seed 42, human weight 4, batch 4, and the same development validation split. Compare per-class AP50, human TP/FP, animal FP, and FPS against the earlier weight-4 baseline. Losses at different input sizes are not directly comparable. A 512-pixel full-image run is the simpler scale check; tiled training is the next experiment if preserving native detail justifies its compute cost.

```powershell
py -3.12 -u -m thermalscout.train_detector --full --data data/processed/yolo_real --out training_runs/NEW_512_RUN --size 512 --epochs 10 --batch 4 --seed 42 --human-weight 4 --device cuda --patience 5
py -3.12 -m thermalscout.evaluate_detector --checkpoint training_runs/NEW_512_RUN/thermal_scout_real_weights.pt --data data/processed/yolo_real --out training_runs/NEW_512_RUN/evaluation.json --device cuda

py -3.12 -u -m thermalscout.train_detector --full --tile --data data/processed/yolo_real --out training_runs/NEW_TILE_RUN --size 256 --epochs 10 --batch 4 --seed 42 --human-weight 4 --device cuda --patience 5
py -3.12 -m thermalscout.evaluate_detector --checkpoint training_runs/NEW_TILE_RUN/thermal_scout_real_weights.pt --data data/processed/yolo_real --out training_runs/NEW_TILE_RUN/evaluation.json --device cuda
```

To find cases worth human review:

```powershell
py -3.12 -m scripts.rank_training_frames --checkpoint training_runs/20261112_025045_full/thermal_scout_real_weights.pt --data data/processed/yolo_real --out training_runs/review_queue.csv --device cuda
```

After reviewing a separate collection of empty training-video frames, place copies in a local folder and pass `--verified-backgrounds "PATH_TO_REVIEWED_EMPTY_FRAMES"` on a new training run. Use an equal-budget control without those frames. Keep adjacent duplicates out of the review set. Preserve the best checkpoint, full run configuration, video split, and evaluation JSON for each experiment.

For a reference architecture, fine-tune a compact pretrained detector on the *same* train and validation videos, then compare quality and throughput against the custom CNN. That is a separate baseline experiment; these options keep ThermalScout-Net's architecture intact.
