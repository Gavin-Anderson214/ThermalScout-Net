
from pathlib import Path

import torch

from thermalscout.model import ThermalScoutReal
from thermalscout.data import BirdsaiDataset
from thermalscout.inference import infer
from thermalscout.evaluate_detector import (
    iou_xyxy,
    xywh_to_xyxy,
)

project = Path(__file__).resolve().parents[1]

checkpoint = (
    project
    / "training_runs"
    / "bambi_backup_gpu_pilot_v2"
    / "thermal_scout_real_weights.pt"
)

data = project / "data" / "bambi_pilot"

device = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(
    checkpoint,
    map_location=device,
    weights_only=True
)

model = ThermalScoutReal().to(device)
model.load_state_dict(ck["model"])
model.eval()

size = int(ck.get("image_size", 512))

dataset = BirdsaiDataset(
    data, "val", size, 0, 42, augment=False
)

print("Device:", device)
print("Validation images:", len(dataset))

# Inspect the first five validation images.
for index in range(min(5, len(dataset))):
    image, ground_truth, path = dataset[index]

    predictions = infer(
        model, image, device, threshold=0.001
    )

    print("\nImage:", Path(path).name)
    print("Ground-truth boxes:", len(ground_truth))
    print("Predictions:", len(predictions))

    if predictions:
        print(
            "Highest confidence:",
            round(predictions[0][1], 4)
        )

    # Find the highest geometric overlap.
    best_iou = 0.0

    for pred_class, score, pred_box in predictions:
        for label in ground_truth:
            true_class, true_box = xywh_to_xyxy(label)

            if pred_class == true_class:
                overlap = iou_xyxy(
                    pred_box, true_box
                )
                best_iou = max(best_iou, overlap)

    print("Best same-class IoU:", round(best_iou, 4))


