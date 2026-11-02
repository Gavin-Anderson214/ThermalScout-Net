"""Keep local data and credentials out of exported research packages."""
from scripts.package_project import shareable


def test_export_excludes_local_assets_even_if_tracked():
    for path in ['data/processed/yolo_real/images/train/frame.jpg',
                 'training_runs/run/model.pt', 'training_runs/run/evaluation.json',
                 'records/experiments/run/predictions.json',
                 'records/experiments/run/human_examples.png', '.env.local',
                 'private.key', '.git/config', '.venv/bin/python', 'dist/old.zip']:
        assert not shareable(path), path


def test_export_keeps_code_evidence_and_navigation():
    for path in ['thermalscout/training.py', 'records/experiments/run/evaluation.json',
                 'records/experiments/run/threshold_tradeoff.png', 'docs/finding.md',
                 'report/Living_Project_Report.html', 'data/README.md',
                 'training_runs/README.md', 'tests/test_core.py']:
        assert shareable(path), path
