# Run your CNN on a thermal image

This command uses the existing custom CNN and a saved checkpoint. It produces an annotated image and a JSON file containing class names, confidence scores and box coordinates in original-image pixels. It does not need labels, retrain the model, or change the network architecture.

From the repository folder, use an existing BIRDSAI-trained checkpoint and a thermal image:

```powershell
$frame = Get-ChildItem data/processed/yolo_real/images/val -File | Sort-Object Name | Select-Object -First 1
py -3.12 -m thermalscout.predict_image --checkpoint training_runs/20261102_classweight_human4/thermal_scout_real_weights.pt --image $frame.FullName --out training_runs/my_image_demo --device auto
Invoke-Item training_runs/my_image_demo/preview.png
```

Choose a new output folder for each run; existing folders are never overwritten. Use `--device cpu` on a machine without CUDA. Use `--threshold 0.5` to inspect a higher confidence cutoff, but that may hide correct detections too. Confidence is the model's score, not a measured probability of being right.

The input is converted to 8-bit grayscale, resized with its aspect ratio preserved, padded with black, and divided by 255 exactly as in the training loader. Predictions are mapped back to the original image dimensions; boxes entirely in padding are discarded. This wrapper uses the existing 500-candidate and 40-detection limits and default inference behavior. It does not calibrate raw thermal sensor temperatures; a future camera feed must provide an image representation compatible with training.

## Verified November 11, 2026

- All 29 project tests passed, including exact preprocessing parity with the training loader on square, landscape and portrait images and box-coordinate round trips.
- A real BIRDSAI validation frame completed inference on CUDA and produced 38 predictions after padding-only boxes were removed. This is a functionality check, not 38 verified objects.
- The existing epoch-7 human-weight-4 checkpoint was reevaluated on all 334 BIRDSAI development-validation frames on CUDA: mAP@50 15.5404%, human true positives 21/62, human false positives 492, animal true positives 623/954 and animal false positives 11,866 at confidence 0.25 and IoU 0.50.
- The result closely reproduces the earlier 15.5420% mAP@50 result. Human counts are unchanged; minor numerical differences remain between evaluations.
- Measured throughput was 17.54 frames/second for inference plus postprocessing on the project desktop GPU, excluding image loading. This is neither end-to-end video throughput nor Raspberry Pi performance.

Full measured evidence: [GPU reevaluation](../records/experiments/20261111_birdsai_gpu_verification/evaluation.json).

## Toward the drone

The next integration step can pass decoded camera frames through `prepare_image`, the existing `infer` function and `original_boxes`. Before claiming live detection, we still need recorded-video testing, Raspberry Pi speed/memory measurements, camera preprocessing checks and better detection quality. Official TestReal remains held out. The current preview is a still-image research demonstration, not a flight demonstration.
