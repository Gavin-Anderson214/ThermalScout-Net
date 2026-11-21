# ThermalScout-Net

**Can a computer find a person or an animal in a grayscale thermal image?** This Tarleton State University COSC 2345 project studies that question using labeled drone images. For each object it finds, the program returns a rectangle, a class (animal or human), and a confidence score. A warm patch alone does not tell the program what made it.

| What would you like to do? | Open |
|---|---|
| Understand one prediction and relate it to class | [Course walkthrough](docs/CLASS_EXPLANATION.md) |
| Open the project on a Pi or Windows PC and run a first example | [Start here](START_HERE.md) |
| Read the Python in order or look up any script | [Code tour and file index](docs/CODE_TOUR.md) |
| See the measurements and their limits | [Development results](docs/RESULTS.md) |

The main detector is a convolutional neural network (CNN) with 88,503 adjustable values called **weights**. During training it sees BIRDSAI frames with answer boxes. During prediction it gets only pixels. The smaller synthetic demo and the pretrained YOLO comparison are separate experiments.

On 334 development frames, the 512-pixel run found 31 of 62 labeled human appearances and drew 53 false human boxes at the recorded cutoff. The human labels came from one video. A person appearing in ten frames can contribute ten appearances. These results describe this data and do not establish live drone performance. [Recorded evaluation](records/experiments/20261112_resolution512_human4/evaluation.json)

**Project credit.** Gavin Anderson's Introduction to AI project at Tarleton State University. ChatGPT generated most or possibly all of the original Python code and assisted with the documentation and teaching comments. The recorded experiments and tests provide evidence readers can inspect. [Course context and acknowledgment](docs/CLASS_EXPLANATION.md)
