# Working folder and presentation handoff

Use the existing `ThermalScout-Net_GitHub_Ready/ThermalScout-Net` repository as the active project. Historical sibling projects are retained as references. Do not train from a duplicate sibling folder.

Start with `docs/BAMBI_REPAIR_2026-11-10.md` for current measured results. Older BIRDSAI reports describe different experiments. The eight-slide presentation is in `report/presentations/ThermalScout-Net_AIAA_2026-11-11.pptx`, with detailed speaker notes and sources.

## Local assets

- Original pilot: `training_runs/bambi_backup_gpu_pilot_v2/thermal_scout_real_weights.pt`.
- Three new experiments: `training_runs/bambi_control_40ep_20261110`, `bambi_hardneg_40ep_20261110`, and `bambi_hardneg_bg1_40ep_20261110`.
- Source data: `data/bambi_source`; prepared pilot: `data/bambi_pilot`.
- Small evidence and hashes: `records/experiments/20261110_bambi_repair`.

No original data, checkpoints or presentations were overwritten. Duplicate cleanup is deliberately limited to an inventory, clearer canonical-folder instructions and exclusions for environments/video/archive files. The duplicate audit does not justify deleting historical copies with potentially different relative-path dependencies. No files were deleted or moved.

## Presentation rehearsal

1. State the goal: a compact custom CNN for thermal human and animal detection.
2. Explain what the model predicts and why brightness alone is insufficient.
3. Describe the 30/30 split and the boar/dog-to-deer distribution shift.
4. Walk through shallow features, deep context, fusion and seven output channels.
5. Explain the repeated confidence clue, the normalization test and the decoder check.
6. Report 0.00%, 1.54%, 4.12% and 4.31% mAP@50 without calling them accuracy.
7. Show the missed-human example and explain why a higher threshold trades recall for fewer false alarms.
8. Close with what is reproducible and what still needs work: diverse human/background training data and an untouched final test.

Likely questions: Why keep the custom CNN? The project studies the original architecture, and tests show it can fit both classes. Why not deploy? Human validation recall is zero and false positives are high. Is this an official benchmark? No, these are custom flight-held-out development scores. Did training simply need more epochs? More epochs alone overfit; data diversity and generalization remain the larger issue. Was the GPU measured in flight? No, experiments ran on the desktop RTX 4070.
