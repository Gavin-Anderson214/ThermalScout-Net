# ThermalScout-Net

**COSC 2345 — Introduction to Artificial Intelligence | Tarleton State University**

Custom computer-vision research project: detect animals and humans in nighttime thermal drone images. The project studies **supervised learning**, **convolutional neural networks (CNNs)**, **transfer learning**, and **model evaluation**. It is **not an LLM**.

## Current status (October 28, 2026)

- **Completed:** custom 29,635-parameter synthetic-data CNN demonstration (8 training epochs, loss 0.9574 → 0.0220).
- **Completed:** inspection of 32 BIRDSAI TrainReal videos and 40,661 thermal frames.
- **Completed:** development export of 2,001 training frames and 334 validation frames, separated by video.
- **Not yet completed on your PC:** training/evaluating the **real-image** custom detector, transfer-learning experiment, or YOLO11 comparison. No real-world accuracy is claimed.
- The `thermalscout/` real-image detector is a separate, more advanced research prototype, not the same 29,635-parameter synthetic CNN.

## Project map

```text
ThermalScout-Net/
├── thermalscout/              # Custom real-image CNN package
│   ├── model.py               # Network layers and prediction head
│   ├── data.py                # Read labeled BIRDSAI frames
│   ├── training.py            # Targets, loss function, learning loops
│   ├── inference.py           # Decode boxes, suppress duplicates, preview
│   ├── train_detector.py      # Full/quick real-image training
│   └── evaluate_detector.py   # Precision, recall, AP/mAP, FPS
├── scripts/birdsai_prepare.py # Inspect and export BIRDSAI training data
├── demo/thermal_scout_demo.py # Small SYNTHETIC teaching demonstration
├── project_manager.py        # Log runs; rebuild living research report
├── records/                  # Real recorded results; absent scores stay blank
├── report/                   # Living Markdown/HTML + previous PDF/Word/Excel
├── docs/                     # Project explanation, progress, GitHub instructions
├── examples/                 # Synthetic screenshot; BIRDSAI preview is local-only
├── data/                     # Your dataset lives here (not pushed to GitHub)
└── training_runs/            # Model weights and new evaluations (not pushed)
```

## Install and verify (Windows PowerShell)

From the repository root in VS Code:

```powershell
python3.13 -m pip install -r requirements-dev.txt
python3.13 -m pytest -q
```

If PyTorch already works on your machine, you can use it without reinstalling.

## Workflows

**1 — Original small synthetic CNN (already completed):**

```powershell
python3.13 demo/thermal_scout_demo.py
# To retrain: python3.13 demo/thermal_scout_demo.py --retrain
```

**2 — Prepare the real BIRDSAI data (already completed on your E: drive):**

```powershell
python3.13 scripts/birdsai_prepare.py --data "FULL_PATH_TO_TrainReal" --mode inspect --previews 0
python3.13 scripts/birdsai_prepare.py --data "FULL_PATH_TO_TrainReal" --mode prepare --previews 0
```

**3 — Train the full custom real-image detector (NEXT):**

```powershell
python3.13 project_manager.py train --mode quick
python3.13 project_manager.py evaluate
# Larger development experiment:
python3.13 project_manager.py train --mode full
```

If the data isn't found automatically, provide `--data "FULL_PATH_TO_yolo_real"`. The training and evaluation commands use **development validation** images, not the untouched official TestReal split. A quick run is only a smoke test.

**4 — Update research notes after every milestone:**

```powershell
python3.13 project_manager.py note --title "Experiment completed" --detail "What changed and what I observed" --status Complete
python3.13 project_manager.py update
```

Open `report/Living_Project_Report.html` in your browser. Numerical detection scores appear only after an actual evaluation.

## GitHub publishing

See [docs/GITHUB_SETUP.md](docs/GITHUB_SETUP.md). **Never commit the full dataset, pretrained checkpoints, `.venv`, or `training_runs`**. They are excluded in `.gitignore`.

## Research ethics and data attribution

BIRDSAI provides annotations for **animal=0** and **human=1**. Detection of a person does *not* establish that they are a poacher. Review dataset licensing and credit its creators before publicly sharing extracted images. Follow your course's academic integrity and AI-use acknowledgement policy.
