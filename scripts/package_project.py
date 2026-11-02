"""Build and verify a shareable project ZIP without local datasets or models."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(
        ['git', '-c', f'safe.directory={ROOT.as_posix()}', *args], cwd=ROOT)


def shareable(name):
    p = PurePosixPath(name)
    parts = {part.lower() for part in p.parts}
    if parts & {'.git', '__pycache__', '.pytest_cache', '.venv', 'venv', 'dist',
                'trainreal', 'testreal', 'yolo_real', 'birdsai_step2_output'}:
        return False
    if p.parts[0] in {'data', 'training_runs'}:
        return len(p.parts) == 2 and p.name in {'README.md', '.gitkeep'}
    if p.name.startswith('.env') or p.suffix.lower() in {
        '.pt', '.pth', '.ckpt', '.onnx', '.weights', '.pyc', '.zip', '.pem', '.key'}:
        return False
    if p.name in {'predictions.json', 'validation_examples.png', 'human_examples.png',
                  'human_padding_comparison.png', 'BIRDSAI_annotated_example.png'}:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    out = args.out or ROOT / 'dist' / f'ThermalScout-Net_{datetime.now():%Y-%m-%d}.zip'
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    candidates = git('ls-files', '-z', '--cached', '--others', '--exclude-standard').decode().split('\0')
    names = sorted({n for n in candidates if n and shareable(n) and (ROOT / n).is_file()})
    for required in ['START_HERE.md', 'README.md', 'report/Living_Project_Report.html',
                     'requirements.txt', 'scripts/package_project.py']:
        if required not in names:
            raise RuntimeError(f'Missing handoff file: {required}')
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    source_commit=git('rev-parse', 'HEAD').decode().strip(),
                    source_has_changes=bool(git('status', '--porcelain').strip()),
                    excluded='Datasets, checkpoints, prediction/image caches, environments, .git, ZIPs',
                    files={})
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            data = (ROOT / name).read_bytes()
            manifest['files'][name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            archive.writestr('ThermalScout-Net/' + name, data)
        archive.writestr('ThermalScout-Net/PACKAGE_MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(out) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('ZIP integrity failure')
        for name, info in manifest['files'].items():
            if hashlib.sha256(archive.read('ThermalScout-Net/' + name)).hexdigest() != info['sha256']:
                raise RuntimeError(f'Fingerprint mismatch: {name}')
    checksum = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(out.suffix + '.sha256').write_text(f'{checksum}  {out.name}\n', encoding='utf-8')
    print(f'Created and verified {out}\nFiles: {len(names)} | Bytes: {out.stat().st_size}\nSHA-256: {checksum}')


if __name__ == '__main__':
    main()
