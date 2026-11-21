# Open the project and run your first example

[Project home](README.md) · [How the model works](docs/CLASS_EXPLANATION.md)

## On a Raspberry Pi with the project environment installed

Open Terminal on the Pi. These commands open the project, run a tiny example, and check the installation.

```bash
cd ~/Projects/ThermalScout-Net
code .
.venv/bin/python lessons/lesson01_pixels.py
.venv/bin/python -m pytest
```

## On Windows with Python 3.12

Open PowerShell in the repository folder. Create the virtual environment and install dependencies on your first visit. Use the last two commands on later visits.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe lessons/lesson01_pixels.py
.\.venv\Scripts\python.exe -m pytest
```

The example prints a tensor shape of `[1, 1, 4, 4]`: one image, one brightness channel, four rows and four columns. Try dividing the brightest pixel value (245) by 255 and compare your answer with the program. Tests check software behavior and run without a trained checkpoint.

## Choose what to read next

| Goal | Starting point |
|---|---|
| Follow pixels through a convolution | [Second example](lessons/lessons02_convolution.py) |
| Understand all seven numbers the CNN predicts | [Detection head example](lessons/lesson03_detection_head.py) |
| Find data preparation, training or evaluation code | [Python file index](docs/CODE_TOUR.md) |
| Reproduce a recorded experiment with local data | [Dataset setup](data/README.md) and [experiment library](docs/EXPERIMENTS.md) |
| Use an existing checkpoint | [Image demo](docs/IMAGE_DEMO.md) |
| Find a slide deck, report or historical record | [Presentations](report/presentations/), [reports](report/), [experiment measurements](records/), [folder guide](docs/SHARING_AND_ORGANIZATION.md) |

BIRDSAI images and trained weights are local assets. The small introductory examples work on a fresh clone after dependency installation.

## Review existing results on the Pi

From the project folder, run `.venv/bin/python -m scripts.review_results` to read the two recorded development evaluations. For a per-video breakdown and observed match frequency by confidence bin, add `--cache PATH/TO/predictions.json` using the corresponding saved BIRDSAI development predictions. The script reports video IDs; multiple videos can belong to one flight. The bins describe retained predictions only.

Run `.venv/bin/python -m scripts.frame_quality path/to/grayscale.png` to inspect a saved 8-bit grayscale camera frame. Values describe displayed brightness. For a regression check after edits, run `.venv/bin/python -m pytest`.

The summary computes precision (correct boxes divided by predicted boxes) and recall (correct boxes divided by labeled boxes) at the recorded cutoff. To identify a copy of the 512-pixel training run, add `--checkpoint /path/to/thermal_scout_real_weights.pt`. This reads the file's SHA-256 digest and compares it to the evaluation record without loading the weights.
