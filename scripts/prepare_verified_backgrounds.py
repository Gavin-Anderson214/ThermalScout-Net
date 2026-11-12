"""Copy only human-reviewed empty TrainReal frames into a fresh training folder."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-csv', type=Path, required=True)
    parser.add_argument('--train-images', type=Path, required=True, help='BIRDSAI TrainReal/images')
    parser.add_argument('--data', type=Path, default=Path('data/processed/yolo_real'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    image_root = args.train_images.resolve()
    allowed = set(json.loads((args.data.resolve()/'split_videos.json').read_text(encoding='utf-8'))['train_videos'])
    with args.review_csv.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    approved = [row for row in rows if row['reviewed_empty'].strip().upper() == 'YES']
    if not approved:
        parser.error('No rows marked YES after manual review')
    if args.out.exists():
        parser.error('Choose a fresh output directory')
    verified = []
    for row in approved:
        source = Path(row['image']).resolve()
        if row['video'] not in allowed or source.parent != image_root / row['video'] or not source.is_file():
            parser.error(f'Approved image is outside its TrainReal training video: {source}')
        verified.append((row, source))
    args.out.mkdir(parents=True)
    manifest = []
    for row, source in verified:
        target = args.out / f"{row['video']}__f{int(row['frame']):07d}{source.suffix.lower()}"
        shutil.copy2(source, target)
        manifest.append({'video':row['video'],'frame':row['frame'],
                         'source':str(source),'file':target.name,
                         'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    with (args.out/'manifest.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=('video','frame','source','file','sha256'))
        writer.writeheader()
        writer.writerows(manifest)
    print(f'Copied {len(manifest)} reviewed empty training frames to {args.out}')
    print('Use --verified-backgrounds with this folder in a fresh detector run.')


if __name__ == '__main__':
    main()
