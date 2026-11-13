# Start here

[← Project home](README.md)

## Read, present or build

- **Understand my idea:** [the presentation story](docs/INTRO_AI_DRONE_STORY.md).
- **Refresh the whole project:** [project refresher](docs/PROJECT_REFRESHER.md).
- **See my network:** [model.py](thermalscout/model.py).
- **Check measured performance:** [results](docs/RESULTS.md).
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

To start continuous GPU training with a live PowerShell listing and browser dashboard, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\launchers\START_CONTINUOUS_GPU.ps1
```

Keep that PowerShell window open. It prints completed epochs, training loss, and validation loss. The best checkpoint and `loss_history.csv` are saved after each completed epoch in a new `training_runs` folder. Press Ctrl+C in that PowerShell window to stop training.

## Need something less common?

[Supporting tools](scripts/README.md) · [Windows launchers](scripts/launchers/)

[Experiment library](docs/EXPERIMENTS.md) · [Full folder guide](docs/SHARING_AND_ORGANIZATION.md) · [Detailed course background](docs/PROJECT_REFERENCE.md)
