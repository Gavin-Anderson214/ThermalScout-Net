# Sharing and organization

Work from the ThermalScout-Net folder containing project_manager.py. The portable workspace opens that folder only. Existing code and data paths are preserved.

| Location | Purpose | ZIP / GitHub? |
|---|---|---|
| START_HERE.md, README.md | Navigation, setup, current results | Yes |
| thermalscout/ | Detector implementation | Yes |
| Root experiment scripts | Threshold, boundary and class-weight experiments | Yes |
| scripts/ | Preparation and ZIP builder | Yes |
| docs/ | Findings, experiments, explanations | Yes |
| records/ | Metrics, provenance, histories, charts | Yes, excluding caches |
| report/ | Current living report, figures, older snapshots | Yes |
| demo/, examples/ | Synthetic demonstration | Yes |
| tests/ | Verification | Yes |
| data/processed/yolo_real/ | Prepared BIRDSAI dataset | Local only |
| training_runs/ | Checkpoints, previews and caches | Local only |
| dist/ | Generated ZIPs | Local only |

## Build a handoff

Run from the project root:

```powershell
py -3.12 scripts/package_project.py
```

The ZIP appears under dist/ and contains one ThermalScout-Net/ folder. Git-tracked and non-ignored files are considered, with additional exclusions for datasets, weights, credentials, caches and generated releases. PACKAGE_MANIFEST.json records the source commit, uncommitted status and SHA-256 fingerprints for included files. The builder checks ZIP integrity and fingerprints, then writes a SHA-256 sidecar. Only Python's standard library and Git are needed to package.

Choose a specific ZIP name with `--out dist/ThermalScout-Net_submission.zip`.

The recipient can open START_HERE.md and report/Living_Project_Report.html immediately. For tests or the synthetic demo, install Python 3.12 and requirements-dev.txt. For real-image experiments, obtain/prepare the dataset separately and provide a local checkpoint or train from scratch. The handoff does not bundle trained inference assets. Local image/checkpoint links and machine-specific historical provenance paths are not included assets; relative links to packaged metrics work offline.

After an experiment, rebuild the report, test, commit and push, then build a fresh ZIP. A ZIP is a snapshot and will not update itself.

## Historical material

The original documents retain their contents and locations:

- docs/IEEE_Proposal_Original.pdf: original proposal.
- docs/Previous_One_Page_Status.pdf: earlier status.
- report/Master_Project_Report_2026-10-28.docx and .pdf: earlier report.
- report/ThermalScout_Experiment_Tracker.xlsx: earlier tracker.

Use the living report for current results. Earlier sibling projects ThermalScout_BIRDSAI_Step2 and ThermalScout_Demo stay in place; relevant preparation/demo code is already in this repository. Do not add those entire sibling folders to the handoff. The data and checkpoint directories are preserved locally.
