# ThermalScout-Net as local AI

The video [Local AI Explained: How to Run AI Models on Your Computer](https://www.youtube.com/watch?v=edIHPoWgIKU) explains local AI by following the complete path: a model, local hardware, input data, inference, and the tradeoff between capability and hardware limits.

ThermalScout-Net follows that same path with a model I built for my drone project:

```text
thermal frame → preprocessing → my CNN → boxes and classes → display
```

## 1. The model

`thermalscout/model.py` defines a seven-convolution-layer network with 88,503 trainable parameters. It receives a grayscale thermal image and predicts objectness, box position, box size, and one of two classes: animal or human.

## 2. The local computer

The RTX 4070 is the development computer. It gives fast feedback while training and testing. A saved checkpoint contains the learned weights; the architecture code and checkpoint must agree before inference can run.

The future target is a Raspberry Pi mounted on the drone. GPU speed cannot be used as a Raspberry Pi speed claim, so that stage requires its own timing and memory measurements.

## 3. The input

Training uses labeled BIRDSAI thermal frames. The loader converts each frame to grayscale, keeps its aspect ratio, pads it to 256 × 256, and scales pixels to 0–1. The same preprocessing is used by `predict_image.py`, so a still-image test follows the training path.

## 4. Inference

During inference, the CNN produces a prediction map. `thermalscout/inference.py` converts the map into confidence-scored boxes and applies class-specific non-maximum suppression. `thermalscout/predict_image.py` maps those boxes back to the original image and saves an annotated preview.

```powershell
$frame = Get-ChildItem data/processed/yolo_real/images/val -File | Sort-Object Name | Select-Object -First 1
py -3.12 -m thermalscout.predict_image `
  --checkpoint training_runs/20261102_classweight_human4/thermal_scout_real_weights.pt `
  --image $frame.FullName `
  --out training_runs/local_ai_demo_01 `
  --device auto
```

## 5. What the measurements mean

The existing BIRDSAI checkpoint reached 15.54% mAP@50 on 334 development-validation images. It detected 21 of 62 labeled human boxes and produced 492 false human detections. The pipeline works, but the detector needs more improvement before live deployment.

## 6. The hardware path

```text
Today:   BIRDSAI → GPU training → saved CNN checkpoint → still-image preview
Next:    recorded thermal video → timing and error analysis
Later:   thermal camera → Raspberry Pi → CNN → live display on the drone
```

The Raspberry Pi deployment and live flight demonstration are goals. They are not included in the current measured results.
