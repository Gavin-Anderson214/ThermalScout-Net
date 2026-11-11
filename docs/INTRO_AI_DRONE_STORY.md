# ThermalScout-Net: teaching my drone to see heat

## The idea

I am building a drone, and I want to build its thermal detection model myself. For Intro to AI, I am training a small custom convolutional neural network using BIRDSAI thermal imagery. My longer-term goal is to put that model on a Raspberry Pi and display live detections from a thermal camera on the drone.

## Simple presentation outline

1. **My goal:** build a drone that can highlight people and animals in thermal video.
2. **What the camera sees:** thermal images show temperature-related contrast; a bright shape alone does not tell us what it is.
3. **What I am building:** a custom CNN that predicts boxes, classes and confidence scores.
4. **How it learns:** BIRDSAI images and labeled boxes provide examples; training adjusts the network when its predictions are wrong.
5. **What makes this my project:** I am connecting the model I wrote to the drone I am building, learning both AI and hardware integration.
6. **What works today:** the custom 88,503-parameter network trains on the GPU and produces detections. Detection quality still needs improvement.
7. **What comes next:** improve held-out detection results, measure Raspberry Pi speed and memory, test recorded thermal video, then test camera integration before a flight demonstration.
8. **The final vision:** thermal camera → Raspberry Pi running my CNN → live boxes on a display, carried by my drone.

## Honest status for questions

The Raspberry Pi deployment and live drone demonstration are goals, not completed results. GPU training performance does not establish Raspberry Pi frame rate. BIRDSAI is the main project dataset; BAMBI is a diagnostic study with different data and splits.

The BAMBI controlled experiments reached 4.31% mAP@50 on a small development-validation pilot, with zero human recall and many false positives. These results do not establish deployment readiness. Historical BIRDSAI results are reported separately in the [README](../README.md) and linked experiment records; they should not be combined with BAMBI as one accuracy score.

The local introductory PowerPoint contains the visual version of this story. This text outline can be shared without distributing dataset images or model checkpoints.
