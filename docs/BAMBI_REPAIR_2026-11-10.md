# BAMBI diagnosis and controlled GPU experiments

The 88,503-parameter custom CNN is unchanged. The pipeline now has a source-label audit, reproducible failure diagnosis, configurable hard-negative training and inference ablations, checkpoint provenance, and encoder/decoder regression tests. Detection remains a research problem: **validation human recall is still zero**.

## Measured development results

All new training runs used 40 epochs, seed 42, AdamW, learning rate 0.001, weight decay 0.0001, batch 2, 512×512 inputs, horizontal flips, human class weight 2 and an RTX 4070. Each run selected its checkpoint by its own minimum validation loss. Losses from differently weighted objectives should not be compared directly. GPU kernels can vary slightly across runs.

| Run | Selected epoch | mAP@50 | mAP@50:95 | Animal TP / FP | Human TP / FP |
|---|---:|---:|---:|---:|---:|
| Original two-epoch pilot | 2 | 0.00% | 0.00% | 0 / 229 | 0 / 0 |
| 40-epoch baseline control | 12 | 1.54% | 0.70% | 15 / 1172 | 0 / 13 |
| Hard negatives, background weight 0.25 | 17 | 4.12% | 0.82% | 25 / 1123 | 0 / 52 |
| Hard negatives, background weight 1.0 | 17 | 4.31% | 1.13% | 18 / 987 | 0 / 195 |

TP/FP use confidence 0.25 and IoU 0.50. There are 70 animal and 16 human validation boxes. AP uses a score floor of 0.001, 101-point interpolation, 500 pre-NMS candidates and at most 40 detections/image. The 4.31% model has 1.50% overall precision and 20.93% recall at confidence 0.25. Higher mAP did **not** establish acceptable false-alarm performance.

At confidence 0.50, the background-weight-1 model returns 5 animal TP, 17 animal FP and 16 human FP. Animal recall falls to 7.14%, and human recall stays zero. At 0.70 it returns no boxes. This is a threshold tradeoff, not a repaired human detector. Optional per-class spatial peaks yield 4.51% mAP@50 on the same weights; the default remains disabled to retain historical comparability. No final operating threshold is validated.

## What caused the pilot failure?

1. The original checkpoint reproduces 0 mAP and 229 animal false positives exactly. The 40-box count is the inference cap, not evidence of 40 objects.
2. Outputs are not constant. Mean absolute raw-output differences across the first five validation images are 0.0191–0.0201. Identical rounded confidence maxima do not prove model collapse.
3. BatchNorm running statistics matter after only 30 optimizer steps. Recomputing moments from training images alone, with no gradient updates, recovers 17 animal matches but 1,183 false positives. Calibration is diagnostic only and does not replace the original checkpoint.
4. Encoder/decoder round-trip tests for both classes pass at IoU 0.95. There is no evidence of a box-coordinate or class-channel mismatch in the current pilot.
5. Dense background averaging dilutes sparse confident mistakes. Optional hard-negative mining trains on the top `max(32, 3 × positive_cells)` background losses per batch. A separate background-weight setting enables the recorded comparison.
6. Human learning works on training images: the hard-negative 0.25 checkpoint has 98.14% training human AP@50, versus 0% validation human AP. This is strong evidence of overfitting/domain shift, not evidence of useful human detection on new flights. The background-weight-1 model has 76.53% training human AP.

## BAMBI audit

The [official annotation format](https://github.com/bambi-eco/Dataset/blob/main/docs/annotation-format.md) defines species strings and MOT coordinates. Species numbers in the table are not the raw annotation `class_id`: the latter encodes species, age, sex and visibility. Conversion uses exact species names after removing the parenthetical common name. Human maps to 1; supported animal species, including hybrid pig, map to 0. Unknown and No-animal never silently become animals.

| Local split | Flights | Images | Source species and boxes |
|---|---|---:|---|
| Train | 47 | 30 | Wild boar 141, dog 4, human 16 |
| Validation | 6, 22 | 30 | Red deer 70, human 16 |

All 247 selected labels match source MOT rows to six-decimal rounding. No same-flight overlap or exact duplicate image hashes exists across the 60 prepared images. One source-video extraction per flight was checked at its zero-based frame index. Validation pixels matched exactly; the flight-47 sample differed by at most one grayscale level due to image conversion. The thermal input is the left 1024×1024 half of a 2048×1024 paired video, resized to 512×512.

All three flights have `split: train` in the distributed metadata. Flights 6 and 22 are locally held out for development, **not BAMBI's official validation/test split**. They use different camera instances and dates from flight 47. Training animals are boar/dogs, while validation animals are deer. Every selected image contains labels; this pilot does not measure false alarms over representative empty footage. Distinct flight IDs do not establish complete location independence. There were no target-cell collisions in these runs.

## Reproduce from the repository root

```powershell
py -3.12 -m pip install -r requirements-bambi.txt -r requirements-dev.txt
py -3.12 -m scripts.audit_bambi
py -3.12 -m scripts.diagnose_bambi
py -3.12 -m pytest -q
py -3.12 -m thermalscout.train_detector --full --data data/bambi_pilot --out training_runs/NEW_UNIQUE_RUN --size 512 --epochs 40 --batch 2 --device cuda --objectness hard-negative --background-weight 1
py -3.12 -m thermalscout.evaluate_detector --checkpoint training_runs/NEW_UNIQUE_RUN/thermal_scout_real_weights.pt --data data/bambi_pilot --out training_runs/NEW_UNIQUE_RUN/evaluation.json --device cuda
py -3.12 -m scripts.evaluate_bambi_experiments
```

The last command analyzes the three named 20261110 runs and writes small summaries. Audit and diagnosis require local source videos/labels and the original pilot checkpoint. A fresh GitHub clone needs those local assets restored separately. Checkpoints and datasets are intentionally excluded from Git.

Evidence lives in `records/experiments/20261110_bambi_repair/`: source/image hashes, original-checkpoint hash, dataset manifest, calibration result, three configs and loss histories, metrics, per-flight results and threshold sweeps. All original checkpoints remain untouched. The architecture file retains the pre-existing feature-extraction refactor, which changes no layers or parameters.

## Next experiments and presentation claims

The honest result is a reproducible custom detector with some animal localization and unresolved generalization. More epochs alone increased the training/validation gap. Expand human and representative background examples from training flights, preserve untouched flights for a final test, and compare a reference detector under the same split before considering architecture changes. These single-seed, repeatedly inspected development results do not establish significance or flight readiness. No drone deployment or live camera benchmark was performed.

The AIAA deck uses the original TexanDrone branding and includes speaker notes, the audit, matched experiment settings, and a visible missed-human example. Older BIRDSAI results remain historical and are not directly comparable with BAMBI scores.
