# Explain this project in COSC 2345

**Problem:** Locate humans and animals in thermal imagery from conservation drones at night. Each detection contains a class label and bounding box.

**Artificial intelligence concepts:**
- **Supervised learning:** learn from labeled images rather than handwritten rules.
- **CNNs:** use learned image filters to recognize patterns in thermal scenes.
- **Search / model selection:** test training settings and model designs to compare performance (not BFS/A* search over pixels).
- **Generalization:** separate training and validation videos to reduce near-duplicate image leakage.
- **Evaluation:** calculate precision, recall, mAP, and frames per second rather than claiming success based on a screenshot.
- **Responsible AI:** report false alarms and missed humans; the system does not label a detected person as a criminal.

**What has been demonstrated:** A 29,635-parameter synthetic-frame CNN learned to classify three simulated objects in one example. Synthetic loss fell from 0.9574 to 0.0220 over 8 epochs. This is **not** real drone-detection accuracy.

**Verified BIRDSAI data prepared for development:** 32 training-source video sequences, 40,661 raw frames; 2,001 prepared training images (6,691 animals, 1,090 humans) and 334 video-separated validation images (954 animals, 62 humans). TestReal remains untouched.

**What remains:** train the real-image CNN on prepared data; record class-specific validation scores; evaluate controlled transfer learning and a YOLO11n baseline; present limitations and ethical concerns in the IEEE report.
