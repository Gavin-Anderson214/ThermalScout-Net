# Project folder guide

[Project home](../README.md) · [Setup and first example](../START_HERE.md) · [Python file index](CODE_TOUR.md)

| What you need | Where to go |
|---|---|
| Understand the CNN for COSC 2345 | Start with the [course walkthrough](CLASS_EXPLANATION.md), then follow an image through the [Python source](CODE_TOUR.md). |
| Run a small example | The [lessons](../lessons/) use small inputs. The [earlier synthetic CNN](../demo/thermal_scout_demo.py) and [example images](../examples/) illustrate the project. For measured performance, use the [development results](RESULTS.md). |
| Find the report or slides | Read the [living report](../report/Living_Project_Report.md). Open the [presentation folder](../report/presentations/) for the AIAA slides, or the [class presentation plan](PRESENTATION_PLAN.md) for the course talk. |
| Check an experiment | Use the [experiment guide](EXPERIMENTS.md) for context and the [run records](../records/experiments/) for individual measurements. The [metrics table](../records/model_metrics.csv) and [run history](../records/training_runs.csv) summarize earlier work. |
| Locate data or weights | The [dataset notes](../data/DATASET_INSTRUCTIONS.md) and [checkpoint notes](../training_runs/CHECKPOINT_INSTRUCTIONS.md) explain where local files belong. |
| Find a tool or test | The [Python index](CODE_TOUR.md) describes the scripts and tests. The [Windows shortcuts](../scripts/launchers/) run from the repository root. |
| Share the project | Follow the ZIP instructions below or use the [GitHub guide](GITHUB_SETUP.md). |

Slide 7 uses flight 22, frame 2135 from [The BAMBI Dataset](https://github.com/bambi-eco/Dataset) by Christoph Praschl and coauthors (2026), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The slide shows the extracted thermal half, converted to grayscale, resized and annotated for a side-by-side comparison.

To rebuild the living report on Windows, run `py -3.12 project_manager.py update` from the project root.

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
