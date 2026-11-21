# How a thermal frame becomes a prediction

[Project home](../README.md) · [Python file index](CODE_TOUR.md)

Imagine someone has drawn a rectangle around an animal in a thermal frame. The pixels are the **input**. The rectangle and the word “animal” are the **label**. Together they form one supervised training example. Here is where each part goes.

1. **Read the example.** The loader rescales the image and adjusts its boxes to keep them aligned with the pixels. A box has a class, center, width and height. A center at (0.5, 0.5) means halfway across and halfway down. [Image loader](../thermalscout/data.py)
2. **Look for patterns.** A convolution slides the same small set of weights across the image. Early layers retain fine detail. Deeper layers combine information over a larger area. The network resizes and joins both sets of features so a small object can be described in context. [Network](../thermalscout/model.py)
3. **Read one grid cell.** At 512 × 512 pixels the output grid is 128 × 128. Each cell produces seven numbers: one object-presence score, four box values, and two class scores. The grid cell containing the labeled center learns that object. Two centers in one cell compete for the same prediction slot. [Target construction](../thermalscout/training.py)
4. **Learn from the error.** Choosing animal or human is **classification**. Estimating box coordinates is **regression**. The training code combines errors for presence, class and box into a number called the loss. Backpropagation computes how the weights affect that loss; the optimizer updates them. An epoch is one pass through the training batches. [Training loop](../thermalscout/training.py)
5. **Make a prediction.** With a saved set of weights, the network can process a new frame. Python combines the presence and class scores, applies a cutoff and removes overlapping duplicate boxes for each class. A score ranks a candidate. Interpreting it as a calibrated probability would require additional evidence. [Decoder](../thermalscout/inference.py)
6. **Check against labels.** A box must match the class and overlap a reference box enough to earn credit. Intersection over union (IoU) divides the shared box area by their combined area. At IoU 0.50, at least half of that combined area must overlap. [Evaluator](../thermalscout/evaluate_detector.py)

## Your first walkthrough on the Pi

From the project folder, run each command separately using the project's Python. These four exercises need no downloaded data or trained weights.

| Run | Predict before looking at the output |
|---|---|
| `.venv/bin/python lessons/lesson01_pixels.py` | The entry 245 becomes about 0.961 because the loader divides brightness by 255. The shape is one image, one channel, four rows, four columns. |
| `.venv/bin/python lessons/lessons02_convolution.py` | How many places can a 3 × 3 window fit in a 4 × 4 grid without padding? Two across and two down. The random filter gives four weighted sums. This lesson adds 0.3 before ReLU, which clips negative values to zero. |
| `.venv/bin/python lessons/lesson03_detection_head.py` | The head reads 48 features at one location and outputs seven values there. Its randomly initialized scores have no learned meaning yet. |
| `.venv/bin/python lessons/lesson07_yolo_conversion.py` | Given a rectangle 37 pixels wide in a 1024-pixel image, predict the normalized width: 37 / 1024 ≈ 0.0361. What would a width of 1 mean? |

### Follow one example all the way through

Suppose an annotator marks an animal centered at **(0.30, 0.50)** with width **0.10** and height **0.20**, measured as fractions of the final square image. This is an invented teaching example.

| Where to pause in the code | What happens to this example |
|---|---|
| `data.py → __getitem__` | The input is a grid of brightness values. The annotation is a separate answer used for learning or scoring. Resize and padding must move the box along with the picture. |
| `model.py → features` | For a 512 × 512 input, the shallow branch yields 24 maps of 128 × 128. The deep branch yields 56 maps of 64 × 64. Resizing the latter lets the CNN join all 80 maps position by position and fuse them into 48. A “map” is a grid of responses from one feature channel. |
| `model.py → forward` | The head turns those 48 values into seven per cell. For our example, training assigns the object to column 38, row 64, because 0.30 × 128 = 38.4 and 0.50 × 128 = 64. The target center offsets are 0.4 and 0. The tensor indexes row before column. |
| `training.py → make_targets` | That cell's presence target is 1, its class ID is 0 (animal), and its size targets are log(12.8) and log(25.6), since the box spans 12.8 by 25.6 cells. An empty cell has a presence target of 0. |
| `training.py → train_epoch` | On each batch, the current weights produce a guess. The loss measures presence, position and class errors. Backpropagation computes gradients; AdamW applies the updates. One batch changes the weights, one epoch visits the training batches, and a checkpoint saves a selected set of weights. |
| `inference.py → infer` | Suppose the presence score after sigmoid is 0.80 and the animal share after softmax is 0.75. The candidate score is 0.60. It clears a 0.25 threshold. Overlapping candidates can still be removed as duplicates. |
| `evaluate_detector.py → get_counts` | A retained animal box with IoU 0.60 can match an unused animal label under the 0.50 overlap rule. A human box at exactly the same location cannot earn that animal match. Class score and geometric overlap measure different things. |

### Connect the new lectures to the code

| Course source | How to explain the connection |
|---|---|
| Lecture 11 pp. 2–10, learning and generalization | In class, an instance has features and a category. Here the instance is an image with zero or more objects; each labeled object has a category and a box. The architecture defines a hypothesis space. One checkpoint selects a function from that space. Generalization means it also works on unseen scenes. |
| Lecture 11 pp. 16–24, decision trees | A tree learns which feature to split on using criteria such as information gain. A CNN learns numerical filters by gradients. Both can learn categories from labels. Reading a single CNN filter gives less direct explanation than following a path down a tree. |
| Lecture 11 pp. 31–35, evaluation | This project uses a development validation split. K-fold cross-validation would rotate the held-out fold and train several times. A plot of loss versus epoch tracks one training run; the lecture's learning curve asks what happens as the number of training examples changes. |
| Chapter 14a pp. 3–10, Bayesian networks | A Bayesian network describes random variables, their parents and local conditional probabilities. In the CNN, layers pass numerical features to later layers. Neither layer connections nor learned filter weights are tables of conditional probabilities. |
| Chapter 14b pp. 3–9, inference | In a Bayesian network one can ask for a posterior given evidence and sum over hidden variables. Here “inference” means evaluating a trained function on observed pixels and decoding its output. Neither enumeration nor variable elimination occurs in this forward pass. A score of 0.60 alone does not establish that six of ten such boxes are correct. |

**Teach it back:** If an animal is labeled correctly but its box is in the wrong place, which part of the loss can still be large? **The box regression loss.** If the saved model draws two boxes around that animal and both survive filtering, how many true positives can it earn? **One**, because one annotation can be matched only once. If you run the model with random weights, what can you still test? **Shapes and execution speed**; trained detection results require the learned checkpoint.

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
