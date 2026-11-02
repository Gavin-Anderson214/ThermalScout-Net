# Start Here — ThermalScout-Net

Open this folder in VS Code or use ThermalScout-Net.code-workspace. Run commands from the folder containing project_manager.py.

## Read or present

1. Open [the living report](report/Living_Project_Report.html) for current results.
2. Read [the latest finding](docs/HUMAN_WEIGHT_FINDING_2026-11-02.md).
3. Review [the experiment](docs/HUMAN_WEIGHT_EXPERIMENT_2026-11-02.md) and [evidence](records/experiments/20261102_human_weight/comparison.json).
4. Use [README.md](README.md) for project and course context.

## Work on the project

```powershell
py -3.12 -m pip install -r requirements-dev.txt
py -3.12 -m pytest tests -q
py -3.12 project_manager.py update
```

Real-image experiments need the prepared dataset under data/processed/yolo_real or an explicit --data path. Weights and diagnostic images stay under training_runs. Official TestReal remains held out. This is a research prototype with many false positives.

## Find things

- thermalscout/: detector implementation.
- docs/: findings, experiments and explanations.
- records/: measured evidence and logs.
- report/: current report, figures and historical snapshots.
- scripts/: preparation and ZIP utilities.
- demo/, examples/: synthetic teaching demonstration.
- tests/: verification.
- data/, training_runs/: local data and model outputs.

Root helpers retain their current names and locations. See the [folder map](docs/SHARING_AND_ORGANIZATION.md) for details.

## Share the project

Run `py -3.12 scripts/package_project.py` to build a ZIP under dist/. Read [sharing instructions](docs/SHARING_AND_ORGANIZATION.md) for recipient setup and the historical-file inventory. The Windows shortcuts use Python 3.12.

Datasets, weights, prediction caches and temporary outputs are excluded. Earlier sibling projects and original documents remain in place. The portable workspace opens only this active project.
