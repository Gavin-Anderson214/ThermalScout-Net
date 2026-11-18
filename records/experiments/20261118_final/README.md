# November 18 desktop verification evidence

`selection.json` contains fresh, full-split reevaluations of three saved checkpoints and selects the highest development mAP50. `candidate_0_evaluation.json` is the historical 512px epoch-4 model, `candidate_1_evaluation.json` is the new baseline epoch-5 model, and `candidate_2_evaluation.json` is the new hard-negative epoch-10 model.

The two new ten-epoch CUDA runs use the same architecture, seed, BIRDSAI split, size, batch, class weight, optimizer, learning rate, background weight and augmentation. Only objectness-loss treatment changes. Their configuration files record package versions, commands and source hashes. Loss CSVs and detection-history JSON files preserve every epoch. FN equals GT minus TP; it is explicit in fresh reevaluation files and the hard-negative history.

The historical reference used minimum validation-loss epoch selection. The new runs use maximum validation mAP50. All three final checkpoints were reevaluated with the same scorer. The historical result remains best overall. The hard-negative full-data checkpoint loses human detection despite a successful small-set memorization diagnostic.

`audit.json` compares all 2,335 prepared frames and labels against TrainReal, checks duplicates and preprocessing, and reports source totals. It captures the obsolete E: path from the local YAML before that metadata path was repaired. Original images and labels were unchanged.

`overfit_*` files preserve the 16-image diagnostic histories and summaries. Both diagnostics use the same selected frames and settings described in [the report](../../../docs/DESKTOP_FINAL_2026-11-18.md). The baseline was run before mAP tie-breaking was refined and saved the first mAP maximum; the later hard-negative diagnostic uses minimum per-class precision to break ties. The baseline's final epoch measurements are still in its history. The diagnostic helper gained automatic source/config recording after these runs, so these early diagnostics have no retrospectively claimed run-time source hash.

`thresholds.csv` records fixed-score operating points on the selected checkpoint's cached development predictions. Per-class best-F1 thresholds in `selection.json` are exploratory and selected on validation; they are not unbiased test results. The main metrics and example figures use confidence 0.25.

`example_manifest.json` records deterministic selection, all predictions and missed boxes for four illustrative frames, plus numbered highlights used in the final slides. It contains no image data. Full validation prediction caches, image previews and weights remain in local, ignored `training_runs/` folders. The final PowerPoint embeds the selected visual examples.

These results describe reused development videos. All human validation labels come from one video, flight identities are unavailable, and shared filename prefixes prevent claiming verified flight independence. Official TestReal was not used.
