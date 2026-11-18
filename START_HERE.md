# Start here

[← Project home](README.md)

## Read, present or build

- **Understand my idea:** [the presentation story](docs/INTRO_AI_DRONE_STORY.md).
- **Refresh the whole project:** [project refresher](docs/PROJECT_REFRESHER.md).
- **See my network:** [model.py](thermalscout/model.py).
- **Check measured performance:** [results](docs/RESULTS.md).
- **Use the desktop submission workflow:** [verified data, memorization, measured checkpoint selection](docs/DESKTOP_FINAL_2026-11-18.md).
- **Improve small-object predictions:** [next detector experiments](docs/NEXT_DETECTOR_EXPERIMENTS_2026-11-12.md).
- **Review external thermal data:** [MONET and Wageningen experiment](docs/MONET_WAGENINGEN_EXPERIMENT_2026-11-12.md).
- **Try a saved model:** [image demo instructions](docs/IMAGE_DEMO.md).

## Run the project

Open a terminal in the repository folder. These commands use Python 3.12 on Windows:

```powershell
py -3.12 -m pip install -r requirements-dev.txt
py -3.12 -m pytest tests -q
```

Training additionally requires your locally prepared BIRDSAI data. A new clone does not include images or weights. See [data setup](data/README.md).

```powershell
py -3.12 project_manager.py train --mode quick --data data/processed/yolo_real --device auto
```

A quick run checks the pipeline; it is not a full accuracy experiment. It saves a new run locally under `training_runs`. Official TestReal remains held out.

Use a bounded experiment with detection-based checkpoint selection for the final project:

```powershell
python -m thermalscout.train_detector --full --size 512 --epochs 10 --batch 4 --human-weight 4 --seed 42 --device cuda --select-by map50 --out training_runs/NEW_EXPERIMENT
```

Use a fresh run name. This saves training/validation loss and detection metrics after each epoch. It retains the highest validation-mAP50 checkpoint. A long run or lower training loss does not guarantee better detections.

To watch an existing run without launching another training job:

```powershell
python live_dashboard.py --watch-run training_runs/NEW_EXPERIMENT
```

The browser shows real loss curves and detection measurements. It refreshes as epochs finish. Closing the dashboard does not stop the separate training process.

## Need something less common?

[Supporting tools](scripts/README.md) · [Windows launchers](scripts/launchers/)

[Experiment library](docs/EXPERIMENTS.md) · [Full folder guide](docs/SHARING_AND_ORGANIZATION.md) · [Detailed course background](docs/PROJECT_REFERENCE.md)
