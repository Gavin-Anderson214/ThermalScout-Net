# Project cleanup — November 11, 2026

The entire local Intro to AI project folder was inventoried, including the active repository, historical BIRDSAI preparation tools, synthetic demo, presentations and Python environments. Sibling coursework folders were outside the cleanup scope.

## Measured audit

- 88,070 original files inventoried.
- 48,563 duplicate candidates hashed with SHA-256.
- 5,850 exact-duplicate groups identified; this is not a deletion list.
- 62 files moved to a recoverable archive: three duplicate preparation tools, one historical derivative presentation and 58 generated cache files.
- Zero files permanently deleted; archiving did not reclaim disk space.

All archived files passed checksum verification. All original files remained at their original or recorded archive paths. The current introductory presentation changed size during the audit and was left untouched; all other original files retained their inventoried sizes.

## Organization and preservation

The local project root now has `START_HERE_PROJECT.md` and `Intro_AI_Project.code-workspace`, grouping the active CNN, presentations, BIRDSAI preparation and earlier demo. Search filters reduce noise from datasets and generated files without moving training data.

The local `_Archive_2026-11-11` folder includes a per-file restore manifest and a restore script that verifies hashes and refuses to overwrite existing destinations. Its preview was checked. Detailed inventories stay local because they contain machine-specific paths.

Datasets, labels, checkpoints, environments, unique source code, experiment records and Git history were preserved. The original presentation template and current introductory deck remain available locally. Dataset duplicates were retained to preserve expected paths and experiment provenance.

The training command's help entry point still loaded after cleanup. No new training or accuracy claims result from this organization work. The custom CNN architecture is unchanged.

Unreviewed dashboard, report and experiment edits remain local and are not part of this documentation update. Large datasets, model weights and local archives are not uploaded.
