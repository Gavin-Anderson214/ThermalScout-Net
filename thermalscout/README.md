# My CNN

[← Project home](../README.md)

This is my custom 88,503-parameter detector. It takes a grayscale thermal image and predicts animal/human classes, confidence scores and bounding boxes.

Follow the complete local-AI path in [LOCAL_AI_FLOW.md](../docs/LOCAL_AI_FLOW.md): thermal frame → preprocessing → CNN → boxes → display.

| I want to understand… | Open |
|---|---|
| The neural network I built | [model.py](model.py) |
| How images and labels enter the model | [data.py](data.py) |
| How the network learns from mistakes | [training.py](training.py) |
| How predictions become boxes | [inference.py](inference.py) |
| How to start training | [train_detector.py](train_detector.py) |
| How performance is measured | [evaluate_detector.py](evaluate_detector.py) |
| How to run one thermal image | [predict_image.py](predict_image.py) |

Start with `model.py`, then follow an image through `data.py`, `training.py` and `inference.py`. Keep these package and module names unchanged so imports continue to work.

[Back to Start Here](../START_HERE.md)
