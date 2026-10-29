# ThermalScout-Net
### Custom AI-Based Human and Wildlife Detection Using Thermal Drone Imagery

**Tarleton State University**  
**COSC 2345 — Introduction to Artificial Intelligence**  
**Project Type:** Computer Vision, Deep Learning, and AI Research  
**Development Language:** Python  
**Primary Framework:** PyTorch  
**Project Status:** In Development

---

## 1. Project Overview

ThermalScout-Net is an artificial intelligence system being developed to detect and classify humans and animals in nighttime thermal drone imagery.

The primary objective is to design, implement, train, and evaluate a custom **Convolutional Neural Network (CNN)** capable of recognizing objects under challenging conditions.

Potential real-world applications include wildlife conservation, nighttime environmental monitoring, and assistance with anti-poaching surveillance.

Rather than relying entirely on an existing pretrained AI model, this project investigates how neural networks learn visual patterns and how different model architectures affect detection performance.

The research will compare:

1. A custom CNN trained from scratch.
2. A custom detector incorporating transfer learning.
3. A pretrained YOLO11 object-detection model fine-tuned on thermal images.

The overall research question is:

**Can a custom-designed neural network effectively detect humans and animals in nighttime thermal drone footage while maintaining reasonable accuracy and computational efficiency?**

---

## 2. Connection to COSC 2345: Introduction to AI

This project applies several important concepts covered in the course.

### 2.1 Intelligent Agents

An intelligent agent perceives information from its environment and uses that information to make decisions.

ThermalScout-Net can be described as the perception and decision-making component of an intelligent monitoring system.

Its environment consists of nighttime thermal imagery.

- **Percepts:** Thermal images collected by drones.
- **State representation:** Pixel intensities and the features extracted by the CNN.
- **Actions or outputs:** Predicted object classes, bounding boxes, and confidence scores.
- **Goal:** Correctly identify humans and animals while minimizing missed detections and false alarms.

The current project is an image-analysis system, not an autonomous drone controller. Drone navigation and autonomous movement are outside its present scope.

### 2.2 Problem Formulation

In artificial intelligence, defining a problem involves identifying the inputs, possible outcomes, and desired objective.

For ThermalScout-Net:

**Input:** A thermal image.

**Output:** The location and classification of every detected human or animal.

**Objective:** Find accurate object locations and classifications while keeping computational requirements manageable.

The model learns a function that maps image data to detection predictions.

### 2.3 Data-Driven Artificial Intelligence

COSC 2345 distinguishes symbolic AI from data-driven AI.

Symbolic AI follows explicitly defined rules.

For example:

"If an object has two legs, classify it as a human."

This approach is unreliable because animals, vegetation, and image artifacts can create similar visual patterns.

ThermalScout-Net instead uses **data-driven AI**, where the neural network learns patterns from labeled images.

Instead of manually defining every visual characteristic, training adjusts the model's internal parameters to reduce prediction errors.

### 2.4 Supervised Machine Learning

ThermalScout-Net uses supervised learning.

Each training image has corresponding ground-truth annotations indicating the object class and location.

During training:

1. The CNN processes an input image.
2. The model predicts object classes and locations.
3. Predictions are compared with the ground-truth labels.
4. A loss function measures the prediction error.
5. Backpropagation calculates how the model's weights should change.
6. An optimizer updates the weights to reduce future errors.

This process repeats across multiple training epochs.

An epoch represents one complete pass through the training dataset.

### 2.5 Artificial Neural Networks

A convolutional neural network processes images through layers containing learnable parameters.

Different layers learn visual features, including:

- Edges and boundaries.
- Local brightness and contrast patterns.
- Spatial relationships.
- More complex object features.

The network uses these features to predict the presence and location of objects.

The first ThermalScout-Net demonstration contains **29,635 trainable parameters**.

The real-image detection architecture is a separate, more advanced model.

### 2.6 Search and Optimization

The course introduces uninformed and informed search algorithms, including BFS, DFS, Uniform-Cost Search, Greedy Best-First Search, and A*.

These algorithms explore possible states or paths to find solutions.

ThermalScout-Net does not use A* or BFS to locate humans inside an image. Object locations are predicted by the neural network.

However, the project applies the broader idea of search and optimization when investigating different model configurations.

Possible parameters to investigate include:

- Learning rate.
- Input image resolution.
- Number of training epochs.
- Detection confidence threshold.
- Neural-network architecture.

A hyperparameter search evaluates different configurations to identify promising solutions.

This creates an experimental connection between search concepts and modern machine learning.

---

## 3. Dataset: BIRDSAI

The project uses BIRDSAI, a dataset containing real thermal drone footage collected for wildlife monitoring research.

The images include human and animal bounding-box annotations.

The dataset uses the following class IDs:

| Class ID | Object |
|---|---|
| 0 | Animal |
| 1 | Human |

### 3.1 Original Real Training Dataset

The following information was verified during the dataset inspection:

| Measurement | Count |
|---|---:|
| Real training video sequences | 32 |
| Extracted thermal image files | 40,661 |
| Animal annotation rows | 74,668 |
| Human annotation rows | 12,531 |
| Unusable sequence directories | 0 |

These are dataset counts, not AI detection results.

### 3.2 Prepared Development Dataset

To make the initial experiments manageable, a smaller subset was exported.

| Measurement | Training | Validation |
|---|---:|---:|
| Images | 2,001 | 334 |
| Animal annotations | 6,691 | 954 |
| Human annotations | 1,090 | 62 |
| Video sequences | 26 | 6 |

### 3.3 Why the Dataset Is Split

The training dataset teaches the model.

The validation dataset helps evaluate its development performance and choose model settings.

The official test dataset is reserved for final evaluation.

Because consecutive video frames can look nearly identical, the training and validation datasets are separated by video sequence instead of randomly mixing nearby frames.

This reduces data leakage and helps measure how the model performs on footage it has not previously encountered.

### 3.4 Class Imbalance

The dataset contains considerably more animal annotations than human annotations.

This presents a machine-learning challenge because a network may become better at recognizing the more frequently represented class.

The project will investigate how this imbalance affects human recall and overall detection performance.

---

## 4. How ThermalScout-Net Works

The complete detection system follows this process:

**Thermal Image → Feature Extraction → Feature Processing → Object Predictions → Bounding Boxes → Evaluation**

### Stage 1: Image Input

Thermal images are loaded from the BIRDSAI dataset.

The images are resized and converted into tensors so they can be processed by PyTorch.

### Stage 2: Feature Extraction

The CNN applies convolutional filters to the image.

These filters learn patterns associated with humans, animals, and surrounding terrain.

### Stage 3: Feature Processing

The network processes learned features to distinguish possible objects from background information.

This is important because many targets in aerial thermal images occupy very few pixels.

The full research architecture will investigate techniques for detecting these small objects.

### Stage 4: Object Prediction

The detector estimates:

- Whether an object is present.
- Whether the object is a human or animal.
- The object's bounding-box coordinates.
- A confidence score.

### Stage 5: Detection Processing

The model converts its predictions into object detections.

Overlapping predictions may be filtered to reduce duplicate detections.

### Stage 6: Performance Evaluation

Predictions are compared against the ground-truth annotations to determine how accurately the network identifies objects.

The results are saved for analysis and comparison.

---

## 5. Current Project Accomplishments

### Phase 1 — Custom CNN Proof of Concept

**Status: Completed**

A custom convolutional neural network was implemented in Python using PyTorch.

The initial model was trained from scratch using synthetic thermal images representing simplified human and animal shapes.

Verified results:

| Measurement | Result |
|---|---|
| Trainable parameters | 29,635 |
| Training epochs | 8 |
| Initial training loss | 0.9574 |
| Final training loss | 0.0220 |
| Reduction in training loss | Approximately 97.7% |
| Objects detected in example | 3 of 3 |
| Training device | CPU |

The training loss decreased as the model learned patterns in the synthetic examples.

These results demonstrate the initial network's learning capability but do not represent real-world detection accuracy.

### Phase 2 — Real Dataset Preparation

**Status: Completed**

The BIRDSAI dataset was downloaded, extracted, and inspected.

The program verified the training video sequences, image files, and annotated frame numbers.

A development dataset was exported into a YOLO-compatible structure.

The original dataset files remain unchanged.

### Phase 3 — Real-Image CNN Training

**Status: Next Development Phase**

The next objective is to train the custom object detector using actual BIRDSAI thermal images.

The project will record training loss, validation performance, inference speed, and sample predictions.

### Phase 4 — Model Comparison

**Status: Planned**

The custom detector will be compared with transfer-learning approaches and YOLO11 using the same evaluation criteria.

### Phase 5 — Final Engineering Report

**Status: In Progress**

The final report will document the methodology, experiments, numerical results, model limitations, and conclusions using an IEEE-style engineering format.

---

## 6. Project Folder Organization

The repository separates different functions to keep the code manageable.

| Folder or File | Purpose |
|---|---|
| `thermalscout/model.py` | Defines the custom CNN architecture |
| `thermalscout/data.py` | Loads and processes thermal images |
| `thermalscout/training.py` | Implements training and loss calculations |
| `thermalscout/inference.py` | Converts predictions into object detections |
| `thermalscout/train_detector.py` | Runs training experiments |
| `thermalscout/evaluate_detector.py` | Calculates evaluation metrics |
| `scripts/birdsai_prepare.py` | Inspects and prepares the BIRDSAI dataset |
| `demo/thermal_scout_demo.py` | Runs the original synthetic CNN |
| `project_manager.py` | Manages experiments and generates reports |
| `records/` | Stores project history and numerical measurements |
| `report/` | Contains the living research report |
| `docs/` | Contains technical explanations and project history |
| `tests/` | Contains automated code checks |
| `training_runs/` | Stores generated training outputs locally |

The full dataset, trained checkpoints, and generated training runs are excluded from GitHub.

---

## 7. Installation and Setup

### Step 1: Open the Project

Open Visual Studio Code.

Select:

**File → Open Folder**

Choose your extracted `ThermalScout-Net` repository folder.

Make sure `README.md` and `project_manager.py` appear in the main folder.

### Step 2: Open PowerShell

Inside VS Code, select:

**Terminal → New Terminal**

Ensure that PowerShell is selected.

### Step 3: Verify Python

Run:

```powershell
python3.13 --version
```

The project has been developed using Python 3.13.

### Step 4: Install the Dependencies

Run:

```powershell
python3.13 -m pip install -r requirements-dev.txt
```

This installs the required packages, including the development testing tools.

### Step 5: Run the Automated Tests

```powershell
python3.13 -m pytest -q
```

These tests check important components of the project, including data processing and neural-network operations.

Passing the tests confirms that the checked code components work as expected. It does not establish detection accuracy.

---

## 8. How to Run the Project

### Experiment 1: Original Synthetic CNN

To run the completed proof of concept:

```powershell
python3.13 demo/thermal_scout_demo.py
```

To retrain the network from scratch:

```powershell
python3.13 demo/thermal_scout_demo.py --retrain
```

This demonstration uses synthetic images and is separate from the full real-image detector.

### Experiment 2: Dataset Preparation

The BIRDSAI preparation phase has already been completed.

If the dataset must be reprocessed, run:

```powershell
python3.13 scripts/birdsai_prepare.py --data "FULL_PATH_TO_TrainReal" --mode inspect --previews 0
```

Then:

```powershell
python3.13 scripts/birdsai_prepare.py --data "FULL_PATH_TO_TrainReal" --mode prepare --previews 0
```

Replace `FULL_PATH_TO_TrainReal` with the actual location of the extracted dataset.

The `--previews 0` option skips preview-image generation.

### Experiment 3: Train the Custom Real-Image Detector

Start with a quick development experiment:

```powershell
python3.13 project_manager.py train --mode quick
```

Evaluate the trained model:

```powershell
python3.13 project_manager.py evaluate
```

The quick experiment is intended to verify that training and evaluation work correctly.

For a larger experiment:

```powershell
python3.13 project_manager.py train --mode full
```

If the prepared dataset cannot be located automatically, specify its location using:

```powershell
python3.13 project_manager.py train --mode quick --data "FULL_PATH_TO_yolo_real"
```

Training results will be stored locally and recorded by the project management system.

---

## 9. How Model Accuracy Will Be Measured

The project uses multiple performance metrics because no single measurement fully describes object-detection quality.

### Precision

Precision measures how many predicted detections are correct.

High precision means fewer false positive detections.

### Recall

Recall measures how many actual objects the model successfully detects.

High recall means fewer objects are missed.

Human recall is particularly important for this application.

### Mean Average Precision (mAP)

mAP evaluates detection quality across the object classes using confidence-ranked predictions and bounding-box overlap.

The project will record mAP@0.50 and mAP@0.50:0.95.

### Inference Speed

Inference speed measures how quickly a trained model processes images.

This may be reported in frames per second (FPS).

### Training Loss

Training loss measures how closely the model's predictions fit the training targets.

A decreasing loss indicates progress in fitting training data, but does not guarantee good validation or test performance.

---

## 10. Planned Model Comparison

The project will compare the following approaches:

| Model | Training Method | Purpose |
|---|---|---|
| Custom ThermalScout-Net | From scratch | Evaluate the custom detector |
| ThermalScout-Net with pretrained features | Transfer learning | Evaluate pretrained feature extraction |
| YOLO11n | Fine-tuning | Compare against an established detector |

Detection results will be added only after the corresponding experiments have been completed.

The comparison will consider detection precision, recall, mAP, inference speed, and computational requirements.

---

## 11. Research Documentation and Progress Tracking

This repository contains a living engineering report that records project accomplishments and experimental results.

After completing a milestone, add a progress note:

```powershell
python3.13 project_manager.py note --title "Experiment completed" --detail "Describe what was tested and observed" --status Complete
```

Update the report:

```powershell
python3.13 project_manager.py update
```

The latest generated report can be opened at:

`report/Living_Project_Report.html`

The project records completed milestones, training results, evaluation metrics, and documented observations.

These reports update when the commands are executed; they are not automatically synchronized in the background.

---

## 12. Limitations and Responsible AI

Thermal object detection presents several challenges:

- Humans may occupy only a small number of pixels.
- Thermal contrast can change with environmental conditions.
- Animals and humans may appear similar from above.
- The training dataset contains unequal numbers of human and animal annotations.
- Detection confidence does not guarantee correctness.

The current project is a research prototype, not a verified field-deployment system.

Additionally, detecting a human in a protected area does not establish illegal activity.

The system is intended to assist human review rather than independently identify suspected poachers.

---

## 13. Expected Final Deliverables

The completed project aims to provide:

1. A working custom CNN-based object detector.
2. A documented real-world thermal image dataset preparation pipeline.
3. Training and validation results.
4. Comparison against pretrained detection methods.
5. Graphs showing learning behavior and detection performance.
6. Sample annotated thermal images.
7. An IEEE-style engineering research report.
8. A classroom presentation explaining the methods and findings.

---

## 14. Project Conclusion

ThermalScout-Net demonstrates how concepts from Introduction to Artificial Intelligence can be applied to a real-world engineering problem.

The project begins with supervised learning and neural-network fundamentals, advances to image-based perception and object detection, and concludes with the experimental evaluation of different AI architectures.

By designing and training a custom model and comparing it with established methods, the project aims to develop a better understanding of how modern AI systems learn, make predictions, and handle uncertain visual information.

**Current Research Status:** Synthetic CNN completed, BIRDSAI dataset prepared, and real-image detector training and evaluation in progress.

**Research Focus:** Custom neural networks, computer vision, transfer learning, model evaluation, and responsible AI.

---

*This project uses the BIRDSAI dataset for educational research. Dataset attribution, licensing conditions, and course academic-integrity requirements should be followed.*