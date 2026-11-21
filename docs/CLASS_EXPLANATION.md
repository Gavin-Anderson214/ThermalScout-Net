# How a thermal frame becomes a prediction

[Project home](../README.md) · [Python file index](CODE_TOUR.md)

Imagine someone has drawn a rectangle around an animal in a thermal frame. The pixels are the **input**. The rectangle and the word “animal” are the **label**. Together they form one supervised training example. Here is where each part goes.

1. **Read the example.** The loader rescales the image and adjusts its boxes to keep them aligned with the pixels. A box has a class, center, width and height. A center at (0.5, 0.5) means halfway across and halfway down. [Image loader](../thermalscout/data.py)
2. **Look for patterns.** A convolution slides the same small set of weights across the image. Early layers retain fine detail. Deeper layers combine information over a larger area. The network resizes and joins both sets of features so a small object can be described in context. [Network](../thermalscout/model.py)
3. **Read one grid cell.** At 512 × 512 pixels the output grid is 128 × 128. Each cell produces seven numbers: one object-presence score, four box values, and two class scores. The grid cell containing the labeled center learns that object. Two centers in one cell compete for the same prediction slot. [Target construction](../thermalscout/training.py)
4. **Learn from the error.** Choosing animal or human is **classification**. Estimating box coordinates is **regression**. The training code combines errors for presence, class and box into a number called the loss. Backpropagation computes how the weights affect that loss; the optimizer updates them. An epoch is one pass through the training batches. [Training loop](../thermalscout/training.py)
5. **Make a prediction.** With a saved set of weights, the network can process a new frame. Python combines the presence and class scores, applies a cutoff and removes overlapping duplicate boxes for each class. A score ranks a candidate. Interpreting it as a calibrated probability would require additional evidence. [Decoder](../thermalscout/inference.py)
6. **Check against labels.** A box must match the class and overlap a reference box enough to earn credit. Intersection over union (IoU) divides the shared box area by their combined area. At IoU 0.50, at least half of that combined area must overlap. [Evaluator](../thermalscout/evaluate_detector.py)

## Read one result yourself

At the recorded cutoff, 31 human boxes matched, 31 labeled appearances were missed, and 53 predicted human boxes were false alarms. **Recall** asks how many labeled appearances the model found: 31 / 62 = 50%. **Precision** asks how many predicted human boxes were correct: 31 / (31 + 53) ≈ 37%. The overall **29.11% mAP@50** averages the classes' average precision at IoU 0.50 across ranked detections. It answers a different question from either single-cutoff percentage. [Experiment evidence](../records/experiments/20261112_resolution512_human4/evaluation.json)

## Where this fits in COSC 2345

Page references below use the supplied Fall 2026 lecture PDFs by Kushal Vangara. Lecture 9 p. 5 is a preview of later material. The implementation details above come from this repository.

| Lecture and PDF pages | Connection you can explain |
|---|---|
| 0, pp. 15, 26 | Course project and generative AI expectations. The welcome deck makes permitted use depend on instructor or assignment instructions. |
| 1, pp. 3–5, 18–19 | Recognizing objects is a perception task. The CNN represents patterns with learned weights. |
| 2, pp. 5–9, 13–16 | Data-driven versus rule-based decisions. The filters learn from examples. A developer writes the score cutoff and duplicate-removal rules. Which part would change after another training epoch? |
| 3, pp. 3–6 | The search unit asks how a problem and its performance measure are defined. Here the task is to locate objects and performance depends on both class and position. |
| 4, pp. 7–14, 22–25; 5, pp. 4–11 | BFS, DFS and A* explore states or paths. A forward pass evaluates layers in a fixed order. Searching for a route would be a separate drone problem. |
| 6, pp. 9–12, 46–49 | Minimax and expectimax model game decisions. The detector takes an image and has no opponent, legal moves or game tree. |
| 7, pp. 3–10 | Constraint satisfaction assigns values under constraints. The one-box-per-cell limit is a representation choice; the training program uses a differentiable loss. |
| 8, pp. 10–12, 25–27 | Logic derives conclusions from premises. A confidence cutoff is an explicit condition in Python; it cannot prove an animal is in the picture. |
| 9, pp. 4–5, 11 | The preview names supervised learning, classification, regression and backpropagation. The responsibility discussion matters when a false box or missed person could mislead someone viewing the output. |

## Check your understanding

- Trace one label from the image loader into the loss. Which four values specify where the object is? Where is its class used?
- Suppose a 63rd human label is added for a previously unlabeled person the model missed. Recall becomes 31/63. Why does the denominator for precision stay at 84?
- Open the decoder and find the line that multiplies two scores. Explain which values were learned and which decision a programmer controls.
