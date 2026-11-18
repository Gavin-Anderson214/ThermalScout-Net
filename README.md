# ThermalScout-Net

### A custom CNN for desktop thermal object detection

I am building my own **convolutional neural network (CNN)** to find people and animals in thermal images. **BIRDSAI is my main dataset.** The December 2, 2026 course goal is a functioning, explainable and measured desktop detector. Raspberry Pi and drone deployment are future work.

Built for **COSC 2345 — Introduction to AI, Tarleton State University**.

## Find what you need

| Start here | What you will find |
|---|---|
| [01 · Get started](START_HERE.md) | Setup and the few commands you need |
| [02 · My CNN](thermalscout/README.md) | The network I built and how the code fits together |
| [03 · Presentation](docs/INTRO_AI_DRONE_STORY.md) | The simple story: thermal images → my CNN → my drone |
| [04 · Results](docs/RESULTS.md) | Measured performance and what still needs work |

For the video-style explanation of the project, read [Local AI flow](docs/LOCAL_AI_FLOW.md).

For a full refresher before working on the code or presenting, read the [project refresher](docs/PROJECT_REFRESHER.md).

**Current verification:** [desktop audit, memorization test, controlled experiments and final instructions](docs/DESKTOP_FINAL_2026-11-18.md).

## Where it stands

The **88,503-parameter custom CNN** trains and runs inference on thermal images. The selected 512-pixel checkpoint was reevaluated on November 18 at **29.11% mAP@50** on development validation, finding **31 of 62 human boxes** with **53 false human detections**. It also finds 683/954 animal boxes with 12,561 false animal detections, so practical reliability remains unfinished.

The unchanged architecture passed a 16-image memorization test with hard-negative background training. Two new full-data runs reached 21.69% and 22.98% mAP50, so the historical epoch-4 checkpoint remains selected. All 35 software tests pass. [Measurements, limitations and demonstration](docs/DESKTOP_FINAL_2026-11-18.md).

An earlier 256-pixel BIRDSAI GPU evaluation reached **15.54% mAP@50** on the same development frames. Both are single-seed research results; detection quality still needs improvement.

**Raspberry Pi deployment, thermal-camera integration, drone mounting and flight testing are future work.** The current submission concerns desktop training and inference.

The repository contains code and measured evidence. Large datasets, model checkpoints and local archives stay off GitHub.

[Detailed project reference](docs/PROJECT_REFERENCE.md) · [Experiment library](docs/EXPERIMENTS.md) · [Folder and sharing guide](docs/SHARING_AND_ORGANIZATION.md)
