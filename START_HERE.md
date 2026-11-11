# Start here

[← Project home](README.md)

## Read, present or build

- **Understand my idea:** [the presentation story](docs/INTRO_AI_DRONE_STORY.md).
- **See my network:** [model.py](thermalscout/model.py).
- **Check measured performance:** [results](docs/RESULTS.md).
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

## Need something less common?

[Experiment library](docs/EXPERIMENTS.md) · [Full folder guide](docs/SHARING_AND_ORGANIZATION.md) · [Detailed course background](docs/PROJECT_REFERENCE.md)
