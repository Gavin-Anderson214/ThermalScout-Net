# ThermalScout-Net

### Teaching my drone to see heat

I am building my own **convolutional neural network (CNN)** to find people and animals in thermal images. **BIRDSAI is my main dataset.** My goal is to run this model on a Raspberry Pi mounted on the drone I am building.

Built for **COSC 2345 — Introduction to AI, Tarleton State University**.

## Find what you need

| Start here | What you will find |
|---|---|
| [01 · Get started](START_HERE.md) | Setup and the few commands you need |
| [02 · My CNN](thermalscout/README.md) | The network I built and how the code fits together |
| [03 · Presentation](docs/INTRO_AI_DRONE_STORY.md) | The simple story: thermal images → my CNN → my drone |
| [04 · Results](docs/RESULTS.md) | Measured performance and what still needs work |

For the video-style explanation of the project, read [Local AI flow](docs/LOCAL_AI_FLOW.md).

## Where it stands

The **88,503-parameter custom CNN** trains and runs inference on thermal images. The latest BIRDSAI GPU check reached **15.54% mAP@50** on development validation. It found **21 of 62 human boxes**, with **492 false human detections**. Detection quality still needs improvement.

**Raspberry Pi deployment and live drone detection are the next goals, not completed features.**

The repository contains code and measured evidence. Large datasets, model checkpoints and local archives stay off GitHub.

[Detailed project reference](docs/PROJECT_REFERENCE.md) · [Experiment library](docs/EXPERIMENTS.md) · [Folder and sharing guide](docs/SHARING_AND_ORGANIZATION.md)
