# Results

[← Project home](../README.md)

## Main project: BIRDSAI

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
