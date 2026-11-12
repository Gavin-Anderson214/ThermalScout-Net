# Thermal pretraining experiment

This rebuild adapts Ray Fu's five stages for building a language model to ThermalScout-Net's job of finding humans and animals in nighttime drone images. The detector gets a practice task with pixels first, then learns to draw boxes from labeled examples. [Ray Fu's video](https://www.youtube.com/shorts/TvbtVey8SbA) · [Research behind the hidden-patch idea](https://arxiv.org/abs/2111.06377)

| Stage in the video | What it means for this project |
|---|---|
| Pretraining | Hide some 16 by 16 pixel patches in a thermal image. Train the feature layers to help fill those patches in using the visible parts. The original pixels provide the answers. |
| Data | Use the 2,001 original BIRDSAI training frames and keep 334 validation frames from separate videos for comparison. The images and labels match the earlier weight-4 run byte for byte. |
| Scaling | Try the practice task on 32 frames, check that it produces features the detector can load, then run one full pass on the training frames. |
| Post-training | Start the detector with those visual features and a fresh box prediction head. Train on the labeled boxes for 10 epochs using seed 42 and human weight 4. |
| Evaluation and systems | Save the model, its settings, a detection preview, and scores from the same validation footage used for the earlier model. |

### Measured results

The two models were scored on the same 334 development validation images. Human counts use confidence 0.25 and box overlap IoU 0.50.

| Model | Correct human boxes | False human boxes | Human AP50 | Animal AP50 | Overall mAP50 |
|---|---:|---:|---:|---:|---:|
| Original weight 4 | 21 of 62 | 492 | 7.11% | 23.97% | 15.54% |
| Thermal pretraining | 6 of 62 | 256 | 0.49% | 12.92% | 6.71% |

The rebuild produced fewer human alarms and missed more labeled people. The original scored higher for both humans and animals. The pretraining recipe added one pass over the training pixels, followed by the same ten rounds of labeled training as the original. This comparison measures one run of each recipe, and the human validation labels come from a single video.

### Try it in PowerShell

Open PowerShell and switch to the folder that contains `thermalscout`. On the project computer, use the path below. If you moved the project, use its new path. Check that `Get-Location` prints the project folder before running Python.

```powershell
Set-Location -LiteralPath 'D:\College Work\Intro To AI\Intro to AI project\ThermalScout-Net_GitHub_Ready\ThermalScout-Net'
Get-Location
```

Use a fresh run folder for each repeat. Start with the pilot, which prints the average error on the hidden pixels and ends with `Saved visual features`.

```powershell
$data = 'data/processed/yolo_real'
$run = 'training_runs/thermal_rebuild_repeat_01'
py -3.12 -u -m thermalscout.pretrain_detector --data $data --out "$run/pilot" --limit 32
```

After the pilot succeeds, run the full practice pass, labeled training, and evaluation one command at a time. Fix any error before continuing. Labeled training creates the preview, and evaluation creates `evaluation.json`.

```powershell
py -3.12 -u -m thermalscout.pretrain_detector --data $data --out "$run/pretrain" --epochs 1
py -3.12 -u -m thermalscout.train_detector --full --data $data --out $run --epochs 10 --size 256 --batch 4 --lr .001 --seed 42 --device cpu --objectness baseline --human-weight 4 --pretrained-features "$run/pretrain/thermal_features.pt"
py -3.12 -u -m thermalscout.evaluate_detector --checkpoint "$run/thermal_scout_real_weights.pt" --data $data --out "$run/evaluation.json" --device cpu
```

Open the preview, then read the counts from all the validation frames. In the output, `tp` counts correctly matched human boxes, `fp` counts false human boxes, and `gt` counts the labeled human boxes.

```powershell
Invoke-Item -LiteralPath "$run/real_detection_preview.png"
$scores = Get-Content -LiteralPath "$run/evaluation.json" -Raw | ConvertFrom-Json
$scores.class_metrics.human | Format-List tp,fp,gt
```

[Saved settings and measured results](../records/experiments/20261110_thermal_pretraining/)
