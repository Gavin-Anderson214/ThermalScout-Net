# Code tour — simple explanations

| File | Purpose | Plain-language explanation |
|---|---|---|
| `scripts/birdsai_prepare.py` | Data conversion | Finds thermal frames and CSV labels, checks their frame numbers, and creates a training/validation folder. |
| `thermalscout/data.py` | Image loader | Resizes a real thermal picture; moves the box coordinates to match the new size; gives it to PyTorch. |
| `thermalscout/model.py` | Your network | Passes pixels through convolutions, combines image features, then outputs one possible detection for each grid cell. |
| `thermalscout/training.py` | Learning | Builds correct-answer targets, measures mistakes (loss), and updates network weights. |
| `thermalscout/inference.py` | Prediction | Turns raw network numbers into readable boxes, removes duplicates, and draws an example. |
| `thermalscout/train_detector.py` | Experiment runner | Trains from random weights, logs each epoch, saves best weights and an example image. |
| `thermalscout/evaluate_detector.py` | Accuracy evaluator | Matches predicted boxes to the correct labels and calculates precision, recall, mAP, and speed. |
| `project_manager.py` | Research organizer | Runs experiments, records evidence, and rebuilds the report. |
| `demo/thermal_scout_demo.py` | Earlier learning example | Generates fake thermal shapes and trains the small 29,635-parameter segmentation CNN. It is **not** the trained real-image detector. |

**Data flow:** Original BIRDSAI frames → preprocessing → training/validation images and boxes → CNN → box/class predictions → evaluation → logged results → living report.

**Known limits:** The prototype real detector predicts a maximum of one box in each stride-4 grid cell. Very small or tightly packed targets can be missed. Validation human examples are limited. Detector scores have not yet been measured on the user's computer.

**One code improvement during reorganization:** Prediction candidate selection now keeps the highest-scoring 500 candidate boxes across the whole image, rather than the first 500 in row order. This removes an unintended top-left bias. It does not constitute an accuracy result.
