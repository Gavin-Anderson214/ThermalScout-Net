# Python file index

[Project home](../README.md) · [Lecture walkthrough](CLASS_EXPLANATION.md)

For a first reading, open the first five files below in order. The other sections are lookups for experiments, lessons, and tools. The first four numbered lessons use tiny examples. Later BAMBI lessons use local files and build on earlier outputs. Historical experiment runners can require CUDA, datasets or checkpoints.

## Follow one image

| File | What to look for |
|---|---|
| [data.py](../thermalscout/data.py) | Load an image and its answer boxes, resize them together, and build batches. |
| [model.py](../thermalscout/model.py) | Define the CNN layers and how they transform pixels into seven values per grid cell. |
| [training.py](../thermalscout/training.py) | Translate a label into a grid target, compute errors, and update the weights. |
| [inference.py](../thermalscout/inference.py) | Decode grid values into boxes, rank candidates, and remove duplicates. |
| [evaluate_detector.py](../thermalscout/evaluate_detector.py) | Match predictions to labeled boxes and compute precision, recall and average precision. |
| [predict_image.py](../thermalscout/predict_image.py) | Load a saved checkpoint and express predictions in the original picture’s coordinates. |
| [train_detector.py](../thermalscout/train_detector.py) | Run a quick software check or train on the full prepared development split. |
| [pretrain_detector.py](../thermalscout/pretrain_detector.py) | Teach the detector's feature layers to reconstruct hidden thermal patches. |
| [__init__.py](../thermalscout/__init__.py) | Mark thermalscout as a Python package. |

## Small examples you can run without a dataset

| File | What to look for |
|---|---|
| [lesson01_pixels.py](../lessons/lesson01_pixels.py) | Represent a 4 × 4 brightness image as a tensor. Inspect its shape and two normalized pixels. |
| [lessons02_convolution.py](../lessons/lessons02_convolution.py) | Slide a random 3 × 3 filter across a 4 × 4 input and see what ReLU does to negative values. |
| [lesson03_detection_head.py](../lessons/lesson03_detection_head.py) | Read the seven output values for one cell of a small prediction grid. |
| [lesson04_bounding_box.py](../lessons/lesson04_bounding_box.py) | Convert box center and size into opposite corners, then into pixels. |
| [lesson07_yolo_conversion.py](../lessons/lesson07_yolo_conversion.py) | Convert a pixel rectangle into a normalized YOLO label by hand. |
| [thermal_scout_demo.py](../demo/thermal_scout_demo.py) | Generate synthetic shapes, train the smaller segmentation example, and draw its output. |

## Run and record experiments

| File | What to look for |
|---|---|
| [project_manager.py](../project_manager.py) | Launch runs, register losses and measurements, and rebuild a local report. |
| [live_dashboard.py](../live_dashboard.py) | View training progress in a local browser or inspect a saved run. |

## Prepare data and check labels

| File | What to look for |
|---|---|
| [birdsai_prepare.py](../scripts/birdsai_prepare.py) | ThermalScout-Net Step 2: inspect real BIRDSAI labels, preview, prepare train/val. |
| [audit_bambi.py](../scripts/audit_bambi.py) | Verify pilot labels against released MOT rows, flight isolation and file hashes. |
| [find_empty_frame_candidates.py](../scripts/find_empty_frame_candidates.py) | List possible empty BIRDSAI training frames for manual full-resolution review. |
| [prepare_verified_backgrounds.py](../scripts/prepare_verified_backgrounds.py) | Copy only human-reviewed empty TrainReal frames into a fresh training folder. |
| [prepare_birdsai_people_mix.py](../scripts/prepare_birdsai_people_mix.py) | Replace animal-only training frames with more people from BIRDSAI TrainReal. |
| [prepare_hit_uav_mix.py](../scripts/prepare_hit_uav_mix.py) | Prepare a matched-size BIRDSAI and HIT-UAV development experiment. |
| [rank_training_frames.py](../scripts/rank_training_frames.py) | Rank labeled training frames for human review using missed and false boxes. |

## Compare results and inspect mistakes

| File | What to look for |
|---|---|
| [analyze_validation.py](../scripts/analyze_validation.py) | Inspect a saved detector on development validation; never reads TestReal. |
| [diagnose_bambi.py](../scripts/diagnose_bambi.py) | Reproduce pilot outputs and isolate BatchNorm statistics using train data only. |
| [diagnose_padding.py](../scripts/diagnose_padding.py) | Frozen-cache padding ablation and human-overlap diagnosis; development only. |
| [evaluate_bambi_experiments.py](../scripts/evaluate_bambi_experiments.py) | Save reproducible threshold and per-flight diagnostics for fixed checkpoints. |
| [preview_class_weight.py](../scripts/preview_class_weight.py) | Draw the same human-heavy validation frames for both class-weight models. |
| [run_class_weight_experiment.py](../scripts/run_class_weight_experiment.py) | Run a seed-42, ten-epoch baseline-versus-human-weight experiment. |
| [summarize_class_weight.py](../scripts/summarize_class_weight.py) | Summarize completed class-weight runs and rebuild the living report. |
| [reference_yolo.py](../scripts/reference_yolo.py) | Optional pretrained-detector comparison on the same BIRDSAI development split. |

## Maintain and package the repository

| File | What to look for |
|---|---|
| [audit_project_duplicates.py](../scripts/audit_project_duplicates.py) | Read-only inventory of identical small source files in historical project folders. |
| [package_project.py](../scripts/package_project.py) | Build and verify a shareable project ZIP without local datasets or models. |

## Tests

| File | What to look for |
|---|---|
| [test_bambi_repair.py](../tests/test_bambi_repair.py) | Check label mapping, target decoding, empty images and candidate selection. |
| [test_class_weight.py](../tests/test_class_weight.py) | Compare gradients when the human class receives more weight. |
| [test_core.py](../tests/test_core.py) | Check shapes, loss, labels, overlap and valid boxes on tiny constructed inputs. |
| [test_experiment.py](../tests/test_experiment.py) | Check run registration after relocation and class-specific confidence filtering. |
| [test_packaging.py](../tests/test_packaging.py) | Check which paths are safe to include in the project export. |
| [test_padding_diagnosis.py](../tests/test_padding_diagnosis.py) | Check the resized content area and distinguish box-position errors from class mistakes. |
| [test_predict_image.py](../tests/test_predict_image.py) | Check that training and prediction use the same preprocessing and reverse padding correctly. |
| [test_tiling.py](../tests/test_tiling.py) | Verify tile coverage and clipping of boxes near tile boundaries. |

## BAMBI pilot lessons

| File | What to look for |
|---|---|
| [lesson05_bambi_labels.py](../lessons/lesson05_bambi_labels.py) | Inspect the annotation rows and count labels in BAMBI flight 47. |
| [lesson06_bambi_visualize.py](../lessons/lesson06_bambi_visualize.py) | Open the video and compare boxes with the frame they label. |
| [lesson08_bambi_converter.py](../lessons/lesson08_bambi_converter.py) | Group flight 47 boxes by frame and write one text file per labeled frame. |
| [lesson09_extract_bambi.py](../lessons/lesson09_extract_bambi.py) | Extract selected thermal frames to pair with the converted labels. |
| [lesson10_verify_bambi.py](../lessons/lesson10_verify_bambi.py) | Check that the exported pictures and labels form usable pairs. |
| [lesson11_prepare_bambi_512.py](../lessons/lesson11_prepare_bambi_512.py) | Resize square thermal images to 512 pixels and write previews. |
| [lesson12_prepare_validation.py](../lessons/lesson12_prepare_validation.py) | Prepare a local validation subset from flights 6 and 22. |
| [lesson13_finalize_bambi.py](../lessons/lesson13_finalize_bambi.py) | Assemble flight 47 training images and labels for the pilot. |
| [lesson14_audit_validation.py](../lessons/lesson14_audit_validation.py) | Count images and labels in the validation split and inspect its flight composition. |
| [lesson15_preview_validation.py](../lessons/lesson15_preview_validation.py) | Draw validation boxes on selected images for visual inspection. |
| [lesson16_diagnose_predictions.py](../lessons/lesson16_diagnose_predictions.py) | Load a local pilot checkpoint and inspect same-class overlap on five validation frames. |
