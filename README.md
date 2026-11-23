# ThermalScout-Net

Open this folder in VS Code. The custom CNN is defined in `thermalscout/model.py`. The 512-pixel epoch-4 checkpoint is in `weights/`. Its classes are animal and human.

Install the dependencies in your Python environment with `python -m pip install -r requirements.txt`. To inspect the existing training, evaluation or image prediction options, run the corresponding module with `--help`, for example `python -m thermalscout.train_detector --help`. Training and scored evaluation require a labeled dataset.
