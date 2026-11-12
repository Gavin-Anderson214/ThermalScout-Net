# Results

[← Project home](../README.md)

## Data and tools

The main dataset is **BIRDSAI**, a thermal drone imagery dataset distributed through the Conservation Drones / LILA Science project. The prepared project data uses its labeled animal and human frames; the original source data and large image folders stay local because of size and dataset terms. BAMBI is a separate pilot dataset used for diagnosis, not mixed into the BIRDSAI score below.

The model is implemented in **PyTorch** because it gives the project a direct way to define the custom CNN, train it with CUDA on the RTX 4070, inspect the loss and predictions, and save/reload the learned weights. PyTorch is the framework used to implement and train this model; it is not a pretrained detector replacing the custom architecture.

## New BIRDSAI size experiment: 512 pixels

On November 12, a bounded CUDA run used all 2,001 training frames, 334 development-validation frames, human class weight 4, seed 42, batch size 4 and 512-pixel input. Early stopping ended at epoch 9 and retained the epoch-4 checkpoint. Evaluation on the same 334 original frames measured **29.11% mAP@50**: 31 of 62 human boxes detected with 53 false human boxes; 683 of 954 animal boxes detected with 12,561 false animal boxes at confidence 0.25 and IoU 0.50. Human AP50 was 35.35% and animal AP50 was 22.87%. Animal false alarms remain high. The 256-pixel weight-4 model below reached 15.54% mAP@50, 21 human hits, 492 false human boxes, and 23.97% animal AP50. This is a single-seed development comparison; the human labels come from one validation video.

A separate 256-pixel continuous run was stopped after 97 epochs. Its best checkpoint was epoch 5, and evaluation found 7.15% mAP@50 with 7 human hits. Continuing to 97 epochs did not improve its best validation loss. This run used human weight 2, so it is not an isolated size comparison with the 512-pixel run.

[Size experiment evidence](../records/experiments/20261112_resolution512_human4/evaluation.json) · [Stopped run evidence](../records/experiments/20261112_025045_full/evaluation.json) · [Next controlled experiments](NEXT_DETECTOR_EXPERIMENTS_2026-11-12.md)

## Previous BIRDSAI training: CUDA, 12 epochs

The new `gpu_run1` completed on **CUDA** using the RTX 4070 setup, with batch size 16 and 256-pixel inputs. These results are transcribed from the supplied terminal log, not independently rerun here.

| Measurement | Result |
|---|---:|
| Model parameters | 88,503 |
| Training / development-validation images | 2,001 / 334 |
| Completed epochs | 12 |
| Best validation loss | **0.7075 at epoch 3** |
| Final training loss | 0.2077 |
| Final validation loss | 0.8889 |

The trainer saved epoch 3 as the best checkpoint. Lower loss is better, but loss is not an accuracy percentage. Validation did not beat epoch 3 during the remaining epochs. New detection mAP, precision, and recall have **not** been measured. The 40 detections reported in the preview are a count on one image, not a quality score.

The earlier CPU run's best validation loss was 0.7148. Batch size also changed from 4 to 16, so this comparison cannot establish that the GPU improved accuracy. No elapsed times were reported, so no speedup is claimed.

[Run details and evidence](../records/experiments/gpu_run1/README.md) · [All 12 epochs](../records/experiments/gpu_run1/loss_history.csv)

## Earlier detection evaluation: BIRDSAI

The existing custom CNN with human class weight 4 was checked again on the GPU on November 11, 2026. This was evaluation of an existing checkpoint, not a new training run.

| Measurement | Result |
|---|---:|
| Model parameters | 88,503 |
| Development-validation images | 334 |
| Overall mAP@50 | 15.54% |
| Correct human detections | 21 of 62 labeled boxes |
| False human detections | 492 |
| Human recall | 33.87% |
| Human precision | 4.09% |

Counts use confidence 0.25 and IoU 0.50. These are detections per frame, not a count of unique people. mAP measures detection performance; it is not the percentage of images classified correctly.

**What this means:** the network learns useful patterns, but it still misses people and produces many false alarms. This is a research prototype. The human validation labels come from one video, so this result does not establish general performance across flights.

[Full GPU measurements](../records/experiments/20261111_birdsai_gpu_verification/evaluation.json) · [Original controlled class-weight comparison](HUMAN_WEIGHT_EXPERIMENT_2026-11-02.md)

## Separate diagnostic study: BAMBI

The 60-frame BAMBI pilot used training flight 47 and validation flights 6 and 22, with 247 annotations. Controlled experiments reached 4.31% mAP@50, with zero human validation recall. This is a different dataset and split; do not combine it with the BIRDSAI score.

[BAMBI diagnosis and evidence](BAMBI_REPAIR_2026-11-10.md)

## What is next?

Improve detection quality, verify on more independent footage, test recorded-video inference, then measure performance on the Raspberry Pi and connect the thermal camera. No live flight demonstration or Raspberry Pi frame rate has been verified for this CNN.

[Try the still-image demo](IMAGE_DEMO.md) · [Browse experiment records](EXPERIMENTS.md)
