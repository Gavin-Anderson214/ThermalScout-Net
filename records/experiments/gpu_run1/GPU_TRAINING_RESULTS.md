# BIRDSAI GPU training: gpu_run1

Transcribed from the user-provided Windows terminal log. This run was not independently repeated here. Training completed on CUDA with 88,503 parameters, 256-pixel inputs, batch size 16, baseline objectness, and human class weight 2.0. Initialization was random.

| Measurement | Result |
|---|---:|
| Completed epochs | 12 |
| Training / validation images | 2,001 / 334 |
| Best validation loss | 0.7075 (epoch 3) |
| Final training loss | 0.2077 |
| Final validation loss | 0.8889 |

The trainer confirmed epoch 3 as the saved best checkpoint. The preview reported 40 detections on one image; that count does not measure accuracy. Detection mAP, precision, recall, and elapsed training time were not supplied. Official TestReal remains held out by the training workflow.

The earlier CPU run had best validation loss 0.7148. This run changed batch size from 4 to 16 as well as the execution device, so it is not a controlled test of GPU effects on accuracy. No speedup can be calculated without timing measurements.

[Per-epoch losses](loss_history.csv) · [Structured summary](summary.json)

Weights and the generated image preview remain on the user's Windows machine under `training_runs/gpu_run1/`; neither is included here.
