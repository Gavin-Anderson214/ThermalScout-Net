# Start Here - ThermalScout-Net

Use this repository for active development. Folder labels explain the current structure; no Python modules or datasets were moved.

## Folder navigation

| Folder | Label | What belongs here |
|---|---|---|
| [thermalscout](thermalscout/README.md) | Real-image detector | Network, data loading, training, predictions and evaluation |
| [scripts](scripts/README.md) | Dataset preparation | BIRDSAI inspection and export |
| [demo](demo/README.md) | Synthetic teaching demo | Earlier small CNN demonstration |
| [data](data/README.md) | Dataset storage | Local datasets; external prepared data can stay where it is |
| [training_runs](training_runs/README.md) | Generated outputs | Weights, histories and evaluations |
| [records](records/README.md) | Research evidence | Metrics, run logs, milestones and dataset audit |
| [report](report/README.md) | Deliverables | Living report, figures and earlier document snapshots |
| [docs](docs/README.md) | Explanations | Code tour, course explanation and planning |
| [examples](examples/README.md) | Illustrations | Supplied example images |
| [tests](tests/README.md) | Verification | Core behavior checks |

## Important root files

- [README.md](README.md): full project description and workflow.
- [project_manager.py](project_manager.py): runs experiments and rebuilds research reports.
- requirements.txt and requirements-dev.txt: runtime and development dependencies.
- pyproject.toml: test configuration.
- RUN_QUICK_AND_UPDATE.bat, EVALUATE_AND_UPDATE.bat, UPDATE_REPORT.bat: existing Windows shortcuts. Read them before running; they can perform experiments or update outputs.

## VS Code

1. Choose **File > Open Workspace from File** and select `ThermalScout-Net.code-workspace` beside this guide.
2. Explorer shows the labeled **01 - ThermalScout-Net (active project)** root. The other labeled roots show the existing dataset output and earlier preparation/demo projects.
3. Open this file and press **Ctrl+Shift+V** for Markdown preview. Each main folder has its own README label.
4. Use **Ctrl+P** to find a filename; right-click the active project root and choose **Open in Integrated Terminal** before running project commands.
5. Read the main README and docs/CODE_TOUR.md, verify your environment, then follow the existing training/evaluation workflow.

Workspace names are display labels only. The Python package names, file paths and import structure remain unchanged. Opening the workspace does not run training.

## Other folders in the original screenshot

- `birdsai_step2_output`: existing dataset audit, previews and prepared exports. Keep in place; inspect the existing yolo_real export before reusing it.
- `ThermalScout_BIRDSAI_Step2`: earlier data preparation project. The active repository also contains scripts/birdsai_prepare.py.
- `ThermalScout_Demo`: earlier synthetic prototype. The active repository also contains demo/thermal_scout_demo.py.
- `ThermalScout-Net_GitHub_Ready`: delivery wrapper containing this active repository.

Do not delete earlier projects or duplicate the dataset just to simplify navigation. The workspace uses relative paths to these existing sibling directories. If you relocate the project, update those workspace paths. This navigation cleanup makes no claim about new training or accuracy results.
