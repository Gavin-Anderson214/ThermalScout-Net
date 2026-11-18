# MONET and Wageningen thermal data experiment

This was a controlled **development** comparison on the existing BIRDSAI video-held-out validation set. Official BIRDSAI TestReal was not used. All runs used the custom 88,503-parameter detector, random initialization, image size 512, batch 4, seed 42, human weight 4, CUDA, a ten-epoch cap, and early stopping after five epochs without lower validation loss. The saved checkpoint is the epoch with the lowest validation loss.

## Sources and preparation

- [MONET](https://tev.fbk.eu/resources/monet): official **training** partition, `runway_01.zip` only (1,018 frames, 569 person-positive frames). The sampled 200 frames each had a person box and no vehicle or exclude annotation. Its oblique runway view differs from BIRDSAI.
- [Wageningen COW MOTS](https://zenodo.org/records/6370315): official **train** partition only, 200 frames sampled evenly from five training videos. Each 16-bit cow instance mask was reduced to a bounding box and mapped to `animal`. Its imagery has a false-color thermal overlay on a visible-light background; the detector converts it to grayscale.
- Both sources are CC BY 4.0. Source archives and prepared images are local and Git-ignored. `scripts/prepare_monet_wageningen_mix.py` records source checksums and chosen frames in each prepared set's `dataset_manifest.json`.

Each set retained 2,001 training frames by replacing BIRDSAI frames of the **same class**, rather than adding more optimization steps. MONET person frames replaced person-only BIRDSAI frames; cow frames replaced animal-only frames. The 334 BIRDSAI validation images and labels were copied byte for byte; their SHA-256 split digest in every manifest is `70c82f899affe3ca8d7d5f3dbf22273555b30cfa739027b6b33a58c48cb58b75`. The training box totals changed: baseline 6,691 animal / 1,090 human; MONET 6,691 / 910; Wageningen 9,559 / 1,090; combined 9,559 / 910. Equal frame count therefore does **not** imply equal box count.

## Results on BIRDSAI development validation

| Training data | Best epoch | mAP50 | mAP50:95 | Animal AP50 | Human AP50 |
|---|---:|---:|---:|---:|---:|
| Existing BIRDSAI baseline | 4 | 29.11% | 6.35% | 22.87% | 35.35% |
| BIRDSAI + 200 MONET replacements | 5 | 11.42% | 3.00% | 19.39% | 3.45% |
| BIRDSAI + 200 Wageningen replacements | 1 | 7.76% | 1.84% | 9.26% | 6.26% |
| BIRDSAI + both sets of 200 replacements | 3 | 13.48% | 3.42% | 19.46% | 7.50% |

All three replacements reduced measured accuracy in this one-seed comparison. Keep the existing BIRDSAI checkpoint as the recommended model. Lower external-data performance may reflect viewpoint and appearance differences, altered box density, or training variability; this experiment cannot isolate those causes. The baseline also has many animal false positives, so none of these checkpoints is ready for operational deployment.

The loss histories, checkpoint weights, and evaluator JSON are under `training_runs/20261112_monet200_512`, `training_runs/20261112_wageningen200_512`, and `training_runs/20261112_monet200_wageningen200_512`. Results were imported into the living report and records.

## Reproduce locally

Place the official `runway_01.zip` under `data/sources/monet/` and `COW_MOTS_dataset.zip` under `data/sources/wageningen_cow/`. From the repository root:

```powershell
py -3.12 -m scripts.prepare_monet_wageningen_mix --out data/processed/yolo_real_monet200 --monet-count 200
py -3.12 -m scripts.prepare_monet_wageningen_mix --out data/processed/yolo_real_wageningen200 --cow-count 200
py -3.12 -m scripts.prepare_monet_wageningen_mix --out data/processed/yolo_real_monet200_wageningen200 --monet-count 200 --cow-count 200
py -3.12 -m thermalscout.train_detector --full --data data/processed/yolo_real_monet200 --out training_runs/another_monet_run --size 512 --epochs 10 --batch 4 --seed 42 --human-weight 4 --device cuda --patience 5
py -3.12 -m thermalscout.evaluate_detector --checkpoint training_runs/another_monet_run/thermal_scout_real_weights.pt --data data/processed/yolo_real --out training_runs/another_monet_run/evaluation.json --device cuda
```

Use a fresh `--out` for every preparation or training run. For a follow-up, first inspect a small sample of converted boxes, then try lower replacement counts and multiple seeds. Consider training with separate source-aware sampling or thermal-only representations before adopting false-color cow imagery.
