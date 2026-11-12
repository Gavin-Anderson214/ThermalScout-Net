# Thermal training data experiment

This experiment asks whether adding more labeled people during training helps ThermalScout-Net on the project's nighttime wildlife footage. Both new mixes replace the same 200 animal-only frames. They retain the original human-containing frames and footage from all 26 training videos.

| Training mix | Training frames | Human boxes | Animal boxes |
|---|---:|---:|---:|
| Original BIRDSAI | 2,001 | 1,090 | 6,691 |
| BIRDSAI with HIT-UAV | 2,001 | 2,378 | 5,668 |
| BIRDSAI with extra TrainReal frames | 2,001 | 1,462 | 5,668 |

The HIT-UAV mix adds 200 frames from the source's training split. Each frame has at least one human label and comes from a nighttime capture. The selection spans recording dates, flight heights, and camera angles. The preparation script converts the source's Person boxes to the project's human class. [HIT-UAV paper and data description](https://pmc.ncbi.nlm.nih.gov/articles/PMC10119175/) · [Source revision and CC BY 4.0 license](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset/tree/669259659b21a2737ca9f5e4bc81892acbe6ecd8)

The BIRDSAI mix adds 40 unused frames from each of the five TrainReal training videos with human labels. The selection favors frames farther in time from frames already selected. The same people can appear in several frames, so box counts describe labeled instances. [BIRDSAI dataset and license](https://lila.science/datasets/conservationdrones)

All runs use 10 epochs, seed 42, batch size 4, 256-pixel inputs, learning rate 0.001, and human weight 4. Checkpoints are selected by the same validation loss rule. All three evaluations use the same 334 BIRDSAI development validation frames. Image and label bytes match across variants. The fixed confidence cutoff is 0.25 for counts and the bounding box match threshold is IoU 0.50. The official TestReal split remains held out.

### Validation results

| Training mix | Correct human detections | False human detections | Human precision | Human AP50 | Animal AP50 | Overall mAP50 |
|---|---:|---:|---:|---:|---:|---:|
| Original BIRDSAI | 21/62 | 492 | 4.09% | 7.11% | 23.97% | 15.54% |
| BIRDSAI with HIT-UAV | 5/62 | 1,111 | 0.45% | 0.04% | 13.01% | 6.53% |
| BIRDSAI with extra TrainReal frames | 16/62 | 1,333 | 1.19% | 1.50% | 3.83% | 2.66% |

The original dataset gave the highest human and animal AP50 on these validation frames. Both variants used more human labels and fewer animal labels. The experiment measures the effect of each complete mix under the fixed training settings.

The validation-loss rule selected epoch 7 for the original, epoch 7 for the HIT-UAV mix, and epoch 2 for the extra BIRDSAI mix. The human validation labels all come from a single video, so the comparison is specific to this development split and seed. More independent nighttime videos with labeled people would make a stronger follow-up.

### Repeat a run

Open PowerShell in the project root. Build either mix in a fresh folder. The HIT-UAV builder retrieves only the selected source images. The BIRDSAI builder uses the original local TrainReal folder.

```powershell
py -3.12 -m scripts.prepare_hit_uav_mix --out data/processed/yolo_hit_uav_repeat

$raw = '..\..\ThermalScout_BIRDSAI_Step2\BIRDSAI_Step2\conservation_drones_train_real\TrainReal'
py -3.12 -m scripts.prepare_birdsai_people_mix --train-real $raw --out data/processed/yolo_birdsai_people_repeat
```

Set `$data` to the mix you built and `$run` to a fresh run folder. Train the model and evaluate against the original validation set.

```powershell
$data = 'data/processed/yolo_birdsai_people_repeat'
$run = 'training_runs/birdsai_people_repeat_01'
py -3.12 -u -m thermalscout.train_detector --full --data $data --out $run --epochs 10 --size 256 --batch 4 --lr .001 --seed 42 --device cpu --objectness baseline --human-weight 4
py -3.12 -m thermalscout.evaluate_detector --checkpoint "$run/thermal_scout_real_weights.pt" --data data/processed/yolo_real --out "$run/evaluation.json" --device cpu
```

Open the run's `evaluation.json`. Under `class_metrics.human`, `tp` counts correct human detections and `fp` counts false human detections. Compare them with the original run's 21 correct and 492 false detections. Check `class_metrics.animal.ap50` for the wildlife result.

[Saved dataset manifests, training settings, and evaluation results](../records/experiments/20261108_dataset_mix/)
