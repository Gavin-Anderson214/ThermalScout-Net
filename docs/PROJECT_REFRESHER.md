# ThermalScout-Net project refresher

This guide is a plain-language tour of the project. Read it before opening individual Python files.

## The project in one sentence

ThermalScout-Net is my own small object-detection CNN that learns from labeled thermal drone images and predicts where animals and humans appear in a frame.

The larger goal is to train and test the model on my desktop, then measure whether it can run on a Raspberry Pi connected to the thermal camera on my drone.

## What problem is the AI solving?

A thermal camera does not give the model a normal color photograph. It gives an image based on infrared heat differences. A bright region might be an animal, a person, vegetation, a warm rock or another object. The model must learn from examples instead of assuming that every bright region is a target.

This is object detection, which has two jobs:

1. Find the object's location with a bounding box.
2. Assign a class to that box.

The current prepared labels use two classes:

- `0 = animal`
- `1 = human`

The model is not currently a species classifier. It does not reliably distinguish hog, deer, dog or coyote as separate classes.

## The datasets

### BIRDSAI: the main project dataset

BIRDSAI is the main dataset used for the Intro to AI project. It contains thermal drone imagery and annotation files describing objects in the frames. The project preparation code converts the source annotations into the YOLO-style format used by the training loader.

BIRDSAI is used because it matches the project goal: detecting wildlife and people from thermal drone footage. The original imagery and large derived folders stay local because they are large and subject to the source dataset's terms. GitHub contains the preparation code, documentation and measured evidence, not the entire image collection.

### BAMBI: a separate diagnostic experiment

BAMBI was used to investigate a training/inference problem. The pilot used 30 training frames from flight 47 and 30 validation frames from flights 6 and 22, with 247 annotations.

BAMBI is not combined with the BIRDSAI score. Its results answer a different diagnostic question and should be reported separately.

## What the important folders mean

| Folder | Purpose |
|---|---|
| `thermalscout/` | The CNN, data loader, training loop, inference code and evaluation code |
| `scripts/` | Dataset preparation, experiment helpers and packaging tools |
| `docs/` | Explanations, experiment writeups and project guides |
| `records/` | Measured metrics, experiment settings and evidence files |
| `report/` | The living project report and figures |
| `tests/` | Small checks that catch broken model, label and coordinate behavior |
| `data/` | Local datasets and prepared images; large folders stay out of GitHub |
| `training_runs/` | Local checkpoints, logs, previews and evaluation outputs |
| `demo/` and `examples/` | Small teaching examples and reference images |

Start with `README.md`, then `thermalscout/model.py`, `thermalscout/data.py`, `thermalscout/training.py`, `thermalscout/inference.py` and `thermalscout/evaluate_detector.py` in that order.

## What each important Python file does

### `model.py`: the CNN design

This file defines `ThermalScoutReal`, the custom seven-convolution-layer network. It has:

- Early layers that preserve fine image detail.
- Deeper layers that learn more complex patterns.
- A feature-fusion step that combines shallow and deep information.
- A detection head that produces seven values per grid cell.

The seven outputs are objectness, x offset, y offset, box width, box height, animal score and human score. The model has 88,503 trainable parameters.

### `data.py`: turning files into training examples

The data loader finds an image and its matching label file. It converts the image to grayscale, resizes it while preserving its aspect ratio, adds black padding to make a square, and scales pixels to the 0–1 range.

The loader applies the same geometric transformation to the labels. This matters because a box must move with the object when the image is resized or padded.

### `training.py`: teaching the CNN

Training starts with randomly initialized weights. For each batch, the model predicts values, the loss function compares those predictions with the labels, and backpropagation adjusts the weights.

The loss includes objectness, box-location, box-size and class terms. The human class weight can be increased in a controlled experiment, but that does not create new human examples. It only changes how strongly human classification mistakes affect the loss.

### `train_detector.py`: running training

This is the command-line entry point. It chooses the data folder, image size, batch size, learning rate, number of epochs, random seed, device and optional training experiment settings.

`--device cuda` uses the RTX 4070 when CUDA is available. A checkpoint stores the model weights and settings needed to reproduce the run.

### `inference.py`: converting raw output into detections

The CNN produces a grid of raw numbers. Inference converts those numbers into normalized bounding boxes, combines objectness with class confidence, removes overlapping duplicates with non-maximum suppression, and limits the number of returned detections.

### `evaluate_detector.py`: measuring results

Evaluation compares predicted boxes with held-out labels. A prediction counts as a match only when it has the correct class and enough overlap with a ground-truth box.

- Precision asks: how many predictions were correct?
- Recall asks: how many labeled objects were found?
- AP measures the precision-recall curve for one class.
- mAP averages AP across the evaluated classes.

The current scores are development results. They are not official TestReal scores and are not a guarantee of field performance.

## Why PyTorch?

PyTorch is the framework used to implement and train the CNN. It gives the project:

- Tensor operations for image and model calculations.
- Automatic differentiation for backpropagation.
- CUDA support for training on the RTX 4070.
- Model saving and loading for later inference.
- A direct way to inspect the model instead of hiding everything behind a high-level detector package.

PyTorch does not replace the custom architecture. `model.py` defines the architecture; PyTorch supplies the machinery that executes and trains it.

## What the current BIRDSAI result means

The existing human-weight-4 checkpoint was reevaluated on 334 development-validation images:

- 15.54% mAP@50 overall.
- 21 of 62 labeled human boxes matched.
- 492 false human detections.
- 33.87% human recall.
- 4.09% human precision.

This means the pipeline is working, but the detector is not yet reliable. A high number of false positives means the model often calls something a human when the label comparison says it was wrong. The result is useful evidence for the course project, not a claim that the model is ready for autonomous use.

The BAMBI pilot reached 4.31% mAP@50 in its best controlled experiment, with zero human validation recall. That result is separate because it uses different footage, labels and splits.

## The local-AI deployment path

The video's local-AI idea maps to this project as:

```text
BIRDSAI examples
    ↓
PyTorch training on RTX 4070
    ↓
saved ThermalScout-Net checkpoint
    ↓
thermal camera frame
    ↓
CNN inference on a computer or Raspberry Pi
    ↓
animal/human boxes on a display
```

The current desktop GPU throughput measurement was about 17.54 frames per second for model inference and postprocessing, excluding file loading. That number is not a Raspberry Pi result.

## What is finished and what is next

Finished:

- Custom CNN architecture.
- BIRDSAI preparation and label pipeline.
- CUDA training and evaluation on the RTX 4070.
- Repeatable tests for labels, boxes, loss and preprocessing.
- Still-image inference command.
- GitHub documentation and a simple presentation.

Next:

1. Improve performance on independent thermal footage.
2. Test recorded-video inference.
3. Measure speed, memory and temperature on the Raspberry Pi.
4. Connect the thermal camera to the Pi.
5. Test the full camera-to-display pipeline on a bench.
6. Integrate the payload with the drone only after the AI system is measured and stable.

The project is strongest when it clearly separates what has been measured from what is planned.
