# Experiment Finding: Increasing Human Classification Weight

**Project:** ThermalScout-Net  
**Experiment date:** November 2, 2026

We tested whether giving human labels more weight during training would improve ThermalScout-Net’s human detection. Two models used the same data, random seed, architecture and ten-epoch training budget. Only the human classification weight changed, from 2 to 4.

| Development validation metric | Weight 2 | Weight 4 |
|---|---:|---:|
| Correct human detections | 12 of 62 | 21 of 62 |
| Human recall | 19.35% | 33.87% |
| False human detections | 187 | 492 |
| Human precision | 6.03% | 4.09% |

Increasing the human classification weight helped the model detect nine additional human instances. However, false human detections also increased, reducing human precision. This demonstrates a tradeoff: the model became more sensitive to humans but less reliable when predicting the human class.

These results come from one random seed on the development validation set. They do not establish final test performance. The next experiment will evaluate confidence thresholds to reduce false positives while retaining as much of the improved human recall as possible.

## Supporting evidence

- [Detailed experiment and reproduction instructions](HUMAN_WEIGHT_EXPERIMENT_2026-11-02.md)
- [Measured comparison and checkpoint fingerprints](../records/experiments/20261102_human_weight/comparison.json)
- [Living research report, Section 13](../report/Living_Project_Report.md)
