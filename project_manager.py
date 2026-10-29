"""ThermalScout-Net local research log + reproducible report generator.

Runs ON YOUR COMPUTER when you invoke it. It is not an always-on service.
Only verified model metrics supplied with evidence are displayed as scores.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RECORDS = ROOT / 'records'
REPORT = ROOT / 'report'
FIGURES = REPORT / 'figures'
RUNS = ROOT / 'training_runs'
DATE = '2026-10-28'

DEFAULTS = {
    'project': 'ThermalScout-Net: Custom Thermal Object Detection',
    'course': 'COSC 2345 - Introduction to Artificial Intelligence',
    'dataset': {
        'original_videos': 32, 'original_frames': 40661, 'all_animal_annotations': 74668,
        'all_human_annotations': 12531, 'train_videos': 26, 'val_videos': 6,
        'train_images': 2001, 'val_images': 334,
        'train_animal': 6691, 'train_human': 1090, 'val_animal': 954, 'val_human': 62,
        'test_status': 'Official TestReal held out; not used for training or validation.'
    },
    'synthetic': {
        'parameters': 29635, 'epochs': 8, 'initial_loss': .9574,
        'final_loss': .0220, 'demo_objects_correct': 3,
        'demo_objects_total': 3, 'status': 'Synthetic proof of concept only, not a real-data accuracy score.'
    }
}

MILESTONES = [
    ('2026-10-28', 'Complete', 'Custom CNN prototype', 'Implemented a 29,635-parameter CNN in Python/PyTorch.'),
    ('2026-10-28', 'Complete', 'Synthetic training', 'Eight epochs; training loss 0.9574 to 0.0220; three objects detected in one synthetic demonstration.'),
    ('2026-10-28', 'Complete', 'BIRDSAI dataset inspection', '32 source videos and 40,661 real frames identified; annotations matched to frames.'),
    ('2026-10-28', 'Complete', 'Video-separated development export', '2,001 train images and 334 validation images; 26 train videos and 6 validation videos.'),
    ('2026-10-28', 'Not started', 'Custom real-image training', 'Train a small anchor-free custom detector from random weights.'),
    ('2026-10-28', 'Not started', 'Transfer-learning experiment', 'Compare custom detector with pretrained feature extraction.'),
    ('2026-10-28', 'Not started', 'YOLO11n baseline', 'Fine-tune a pretrained object detector on the same split.'),
    ('2026-10-28', 'Not started', 'Held-out evaluation', 'Compute class-specific precision/recall, mAP@.50, mAP@.50:.95, and FPS.'),
]

FIELDS = ['timestamp', 'model', 'training', 'split', 'precision', 'recall', 'map50', 'map5095', 'fps', 'evidence']
RUN_FIELDS = ['timestamp', 'run_id', 'model', 'mode', 'epochs', 'train_loss_final', 'val_loss_best', 'source_csv', 'sha256']
EVENT_FIELDS = ['date', 'status', 'milestone', 'details']


def read_csv(path: Path):
    """Read an experiment log as rows of named columns."""
    if not path.is_file():
        return []
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, columns, rows):
    """Save experiment records in a spreadsheet-friendly CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in columns})


def initialize():
    """Create default tracking files when starting a new repository."""
    RECORDS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    if not (RECORDS / 'project_facts.json').exists():
        (RECORDS / 'project_facts.json').write_text(json.dumps(DEFAULTS, indent=2), encoding='utf-8')
    for name, cols, initial in [
        ('milestones.csv', EVENT_FIELDS, [dict(zip(EVENT_FIELDS, row)) for row in MILESTONES]),
        ('model_metrics.csv', FIELDS, []),
        ('training_runs.csv', RUN_FIELDS, []),
    ]:
        if not (RECORDS / name).exists():
            write_csv(RECORDS / name, cols, initial)


def get_data_root(explicit=None):
    """Find prepared training images without copying the full dataset."""
    if explicit:
        data = Path(explicit).expanduser().resolve()
        return data if (data / 'images' / 'train').is_dir() else None
    project = ROOT.parent
    candidates = [
        project / 'ThermalScout_BIRDSAI_Step2' / 'BIRDSAI_Step2' / 'birdsai_step2_output' / 'yolo_real',
        project / 'BIRDSAI_Step2' / 'birdsai_step2_output' / 'yolo_real',
        ROOT / 'data' / 'processed' / 'yolo_real',
        ROOT / 'birdsai_step2_output' / 'yolo_real',
    ]
    return next((p for p in candidates if (p/'images'/'train').is_dir() and (p/'labels'/'val').is_dir()), None)


def register_loss_csv(path: Path, mode='development'):
    """Record a training run only once, using a file checksum."""
    if not path.exists():
        raise FileNotFoundError(f'No training history exists at {path}')
    raw = path.read_bytes()
    key = hashlib.sha256(raw).hexdigest()
    known = read_csv(RECORDS / 'training_runs.csv')
    if any(r['sha256'] == key and r['source_csv'] == str(path.resolve()) for r in known):
        return False
    rows = read_csv(path)
    if not rows or not {'epoch', 'train_loss', 'val_loss'} <= set(rows[0]):
        raise ValueError('Expected epoch, train_loss, val_loss columns')
    vals = [float(r['val_loss']) for r in rows]
    stamp = datetime.now().isoformat(timespec='seconds')
    known.append({
        'timestamp': stamp, 'run_id': path.parent.name, 'model': 'ThermalScout-Net (real/scratch)',
        'mode': mode, 'epochs': len(rows), 'train_loss_final': rows[-1]['train_loss'],
        'val_loss_best': min(vals), 'source_csv': str(path.resolve()), 'sha256': key,
    })
    write_csv(RECORDS / 'training_runs.csv', RUN_FIELDS, known)
    return True


def svg_loss_plot(path: Path):
    """Plot training and validation loss so learning can be compared."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        rows = read_csv(path)
        xs = [int(r['epoch']) for r in rows]
        train = [float(r['train_loss']) for r in rows]
        val = [float(r['val_loss']) for r in rows]
        fig, ax = plt.subplots(figsize=(7.5, 3.0), dpi=135)
        ax.plot(xs, train, marker='o', label='Training loss')
        ax.plot(xs, val, marker='o', label='Validation loss')
        ax.set(xlabel='Epoch', ylabel='Loss', title='Real BIRDSAI experiment: loss history')
        ax.legend(loc='best'); ax.grid(alpha=.22); fig.tight_layout()
        FIGURES.mkdir(exist_ok=True, parents=True)
        fig.savefig(FIGURES / 'real_training_loss.png', facecolor='white')
        plt.close(fig)
        return True
    except ImportError:
        return False


def synthetic_loss_plot():
    """Graph the recorded eight-epoch synthetic training history."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        values=[.9574,.3722,.2400,.1815,.1320,.0702,.0278,.0220]
        fig,ax=plt.subplots(figsize=(7.5,2.8),dpi=135)
        ax.plot(range(1,9),values,marker='o')
        ax.set(xlabel='Epoch',ylabel='Training loss',title='Synthetic CNN: measured training loss')
        ax.grid(alpha=.22)
        fig.tight_layout()
        fig.savefig(FIGURES/'synthetic_training_loss.png',facecolor='white')
        plt.close(fig)
        return True
    except ImportError:
        return False


def register_evaluation_json(path:Path, mode='development'):
    """Import measured detector metrics from an evaluator output file."""
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('split') not in ('validation','test'):
        raise ValueError('Expected evaluation split validation/test')
    row={'timestamp':datetime.now().isoformat(timespec='seconds'),
         'model':data['model']+' ('+mode+')', 'training':data.get('training','scratch'),
         'split':data['split'], 'precision':data.get('precision',''),
         'recall':data.get('recall',''), 'map50':data.get('map50',''),
         'map5095':data.get('map5095',''), 'fps':data.get('fps',''),
         'evidence':str(path.resolve())}
    records=read_csv(RECORDS/'model_metrics.csv')
    if not any(r['evidence']==row['evidence'] for r in records):
        records.append(row)
        for cls,vals in data.get('class_metrics',{}).items():
            records.append({**row,'model':row['model']+' — '+cls,
                            'precision':vals.get('precision',''),'recall':vals.get('recall',''),
                            'map50':vals.get('ap50',''),'map5095':vals.get('map5095',''),
                            'fps':'','evidence':row['evidence']+' #class='+cls})
        write_csv(RECORDS/'model_metrics.csv',FIELDS,records)
        return True
    return False


def png_model_chart(metrics):
    """Compare measured mAP scores across different experiment runs."""
    rows = [r for r in metrics if r.get('map50') not in (None, '')]
    if not rows:
        return False
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names = [r['model'][:32] + ' (' + r['split'] + ')' for r in rows]
        values = [100 * float(r['map50']) for r in rows]
        fig, ax = plt.subplots(figsize=(8, max(2.6, .5*len(rows)+1.2)), dpi=135)
        ax.barh(names, values)
        ax.set(xlim=(0,100), xlabel='mAP@0.50 (%)', title='Measured object-detection performance')
        ax.invert_yaxis(); fig.tight_layout()
        fig.savefig(FIGURES / 'model_comparison.png', facecolor='white')
        plt.close(fig)
        return True
    except ImportError:
        return False


def fmt(v, percent=False):
    """Present measured values clearly; leave missing results unmeasured."""
    if v in (None, ''):
        return 'Not measured'
    try:
        return f'{float(v)*100:.1f}%' if percent else f'{float(v):.2f}'
    except (ValueError, TypeError):
        return str(v)


def report_update():
    """Rebuild the research report using logged experiments and milestones."""
    initialize()
    facts = json.loads((RECORDS / 'project_facts.json').read_text(encoding='utf-8'))
    events, metrics, runs = [read_csv(RECORDS / file) for file in
                             ('milestones.csv', 'model_metrics.csv', 'training_runs.csv')]
    d, s = facts['dataset'], facts['synthetic']
    latest_history = Path(runs[-1]['source_csv']) if runs else None
    has_loss = latest_history and latest_history.exists() and svg_loss_plot(latest_history)
    has_chart = png_model_chart(metrics)
    has_synthetic = synthetic_loss_plot()
    status = 'Completed: synthetic CNN and BIRDSAI data preparation. In progress: real-image detector training.'
    lines = [
        '# ThermalScout-Net | Living Research Report',
        f'**Course:** {facts["course"]}  ',
        '**Application:** conservation and potential anti-poaching support through thermal aerial object detection.  ',
        f'**Report rebuilt locally:** {datetime.now().strftime("%Y-%m-%d %H:%M")}  ',
        f'**Status:** {status}',
        '', '## 1. The problem and AI approach',
        'The project designs and tests a custom convolutional neural network (CNN) to locate animals and people in infrared drone footage. '
        'A network learns visual representations from labeled images and predicts bounding boxes and classes. '
        'Research compares training from scratch, transfer learning, and YOLO11n. '
        'Hyperparameter search is a model-selection procedure and should not be confused with BFS or A* pathfinding.',
        '', '## 2. Verified milestones',
        '| Date | Status | Milestone | Evidence / notes |', '|---|---|---|---|',
    ]
    for e in events:
        lines.append('| ' + ' | '.join(str(e.get(k,'')).replace('|','/') for k in EVENT_FIELDS) + ' |')
    lines += ['', '## 3. Dataset and scientific controls',
        f'- Original TrainReal: {d["original_videos"]:,} sequences; {d["original_frames"]:,} images; '
        f'{d["all_animal_annotations"]:,} animal annotations; {d["all_human_annotations"]:,} human annotations.',
        f'- Prepared development split: {d["train_videos"]} train videos ({d["train_images"]:,} images, '
        f'{d["train_animal"]:,} animal and {d["train_human"]:,} human labels); '
        f'{d["val_videos"]} validation videos ({d["val_images"]:,} images, '
        f'{d["val_animal"]:,} animal and {d["val_human"]:,} human labels).',
        '- Videos are split by sequence (not by adjacent image) to reduce data leakage.',
        f'- {d["test_status"]}',
        '- The small validation sample has only 62 human labels: human recall will have high uncertainty.',
        '', '## 4. Custom CNN demonstration (synthetic)',
        f'- Custom trainable parameters: {s["parameters"]:,}. Trained {s["epochs"]} epochs on synthetic shapes.',
        f'- Training loss: {s["initial_loss"]:.4f} -> {s["final_loss"]:.4f}.',
        f'- A single illustrated synthetic scene had {s["demo_objects_correct"]}/{s["demo_objects_total"]} objects detected.',
        '- This is not a real-footage precision, recall, mAP, or generalization result.',
        '', '## 5. Real-image training runs',
        '| Timestamp | Model | Mode | Epochs | Final train loss | Best val loss |',
        '|---|---|---|---:|---:|---:|',
    ]
    for r in runs:
        lines.append(f'| {r["timestamp"]} | {r["model"]} | {r["mode"]} | {r["epochs"]} | {fmt(r["train_loss_final"])} | {fmt(r["val_loss_best"])} |')
    if not runs:
        lines.append('| - | Custom real-image model | Not yet run | - | - | - |')
    if has_synthetic:
        lines += ['', '![Measured synthetic-loss curve](figures/synthetic_training_loss.png)']
    if has_loss:
        lines += ['', '![Real training-loss curve](figures/real_training_loss.png)']
    lines += ['', '## 6. Research comparison: measured detection scores',
        '| Model | Training | Split | Precision | Recall | mAP@.50 | mAP@.50:.95 | FPS | Evidence |',
        '|---|---|---|---:|---:|---:|---:|---:|---|']
    if metrics:
        for r in metrics:
            lines.append('| ' + ' | '.join([
                r['model'], r['training'], r['split'], fmt(r['precision'],True), fmt(r['recall'],True),
                fmt(r['map50'],True), fmt(r['map5095'],True), fmt(r['fps']), r['evidence'].replace('|','/')
            ]) + ' |')
    else:
        for m, tr in [('ThermalScout (real)', 'From scratch'), ('ThermalScout (real)', 'Transfer learning'),
                      ('YOLO11n', 'Transfer learning')]:
            lines.append(f'| {m} | {tr} | Not yet evaluated | - | - | - | - | - | - |')
    if has_chart:
        lines += ['', '![Measured model comparison](figures/model_comparison.png)']
    lines += ['', '**Interpretation:** Lower training loss does not establish higher object-detection accuracy. '
              'Only enter precision, recall, mAP, and FPS after running the same evaluation protocol on an appropriate held-out split.',
             '', '## 7. Risks and limitations',
             '- Human annotations are less common than animal annotations.',
             '- People are often only a few pixels wide; some objects are ambiguous.',
             '- The first custom detector uses one prediction per grid cell and can miss crowded targets.',
             '- Prototype output may show detection boxes before trustworthy benchmark metrics exist.',
             '', '## 8. Next engineering actions',
             '1. Run the packaged custom detector using `python3.13 project_manager.py train --mode quick`.',
             '2. Inspect ground truth versus predictions and review the training/validation losses.',
             '3. Run a larger scratch-training experiment and record its settings and limitations.',
             '4. Implement a class-wise IoU-matched precision/recall/mAP evaluation before comparing models.',
             '5. Train a controlled transfer-learning model and a YOLO11n baseline on the same data.',
             '6. Report unbiased final metrics only after the official test set is evaluated.',
             '', '## 9. Course connection',
             'COSC 2345 covers intelligent systems, search and heuristics, uncertainty, supervised learning, '
             'artificial neural networks, and responsible AI. This project emphasizes supervised learning '
             'and CNN-based perception, while using controlled model selection and evidence-based evaluation.',
             '', '## 10. How this report stays current',
             'Run `python3.13 project_manager.py update` after each experiment or add milestones with '
             '`python3.13 project_manager.py note --title "..." --detail "..."`. '
             'The files in this project folder update when you execute the command; nothing runs in the background.',
             '']
    md = '\n'.join(lines)
    (REPORT / 'Living_Project_Report.md').write_text(md, encoding='utf-8')
    # Simple, offline and dependency-free HTML rendering for the same report.
    import re
    blocks, table_rows, paragraph = [], [], []
    def flush_para():
        if paragraph:
            blocks.append('<p>' + '<br>'.join(paragraph) + '</p>'); paragraph.clear()
    def flush_table():
        if table_rows:
            parsed = []
            for j, cells in enumerate(table_rows):
                tag = 'th' if j == 0 else 'td'
                parsed.append('<tr>' + ''.join(f'<{tag}>'+c+f'</{tag}>' for c in cells) + '</tr>')
            blocks.append('<div class="scroll"><table>' + '\n'.join(parsed) + '</table></div>')
            table_rows.clear()
    def esc_inline(t):
        e = html.escape(t)
        e = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', e)
        e = re.sub(r'`([^`]+)`',r'<code>\1</code>',e)
        return e
    for line in lines:
        if line.startswith('|'):
            flush_para()
            cells = [x.strip() for x in line.strip('|').split('|')]
            if cells and all(re.fullmatch(r'[:\- ]+',x) for x in cells):
                continue
            table_rows.append([esc_inline(c) for c in cells]); continue
        flush_table()
        if line.startswith('!['):
            flush_para()
            m = re.fullmatch(r'!\[(.+)\]\((.+)\)', line)
            if m: blocks.append(f'<figure><img src="{html.escape(m[2])}" alt="{html.escape(m[1])}"><figcaption>{html.escape(m[1])}</figcaption></figure>')
        elif line.startswith('#'):
            flush_para()
            level = min(3, len(line)-len(line.lstrip('#')))
            blocks.append(f'<h{level}>' + esc_inline(line.lstrip('#').strip())+f'</h{level}>')
        elif line.startswith('- ') or re.match(r'^\d+\. ',line):
            flush_para(); blocks.append('<p class="list">' + esc_inline(line) + '</p>')
        elif not line.strip(): flush_para()
        else: paragraph.append(esc_inline(line))
    flush_para(); flush_table()
    css = '''body{font-family:Segoe UI,Arial,sans-serif;color:#233040;max-width:1050px;margin:34px auto;padding:0 28px;line-height:1.53}h1{color:#44276e;font-size:30px;border-bottom:3px solid #7f66ae;padding-bottom:9px}h2{color:#452877;font-size:20px;margin-top:28px}h3{color:#452877}p{font-size:14px} .list{margin:5px 0;padding-left:10px;border-left:3px solid #eee}table{border-collapse:collapse;min-width:750px;width:100%;font-size:12px}th{background:#43276f;color:white;text-align:left}td,th{padding:8px 9px;border:1px solid #dbe1e6}tr:nth-child(even){background:#f7f6fa}.scroll{overflow-x:auto;margin:12px 0}code{background:#f3f0f7;padding:1px 4px}figure{margin:16px 0}figure img{width:100%;max-width:780px}figcaption{font-size:12px;color:#64748b}@media print{body{margin:12mm}h2{break-after:avoid}.scroll{overflow:visible}table{min-width:0}}'''
    doc = '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>ThermalScout-Net Living Project Report</title><style>'+css+'</style></head><body>'+'\n'.join(blocks)+'</body></html>'
    (REPORT / 'Living_Project_Report.html').write_text(doc, encoding='utf-8')
    print('Report updated:', REPORT / 'Living_Project_Report.html')
    print('Markdown source:', REPORT / 'Living_Project_Report.md')
    print('Measured model metrics:', len(metrics), ' | recorded training runs:', len(runs))


def main():
    """Choose a task: train, evaluate, record a note, or update the report."""
    p = argparse.ArgumentParser(description='Organize ThermalScout-Net results and build the living report')
    sub = p.add_subparsers(dest='command', required=True)
    up = sub.add_parser('update', help='Rebuild report; optionally import a training-loss CSV')
    up.add_argument('--loss-csv', type=Path)
    up.add_argument('--mode', default='development')
    up.add_argument('--metrics-json', type=Path, help='Import verified evaluator JSON')
    note = sub.add_parser('note', help='Add a dated progress note')
    note.add_argument('--title', required=True); note.add_argument('--detail', required=True)
    note.add_argument('--status', default='In progress', choices=['Complete','In progress','Not started','Blocked'])
    metric = sub.add_parser('metrics', help='Record measured object-detection metrics with evidence')
    metric.add_argument('--model', required=True); metric.add_argument('--training', required=True)
    metric.add_argument('--split', required=True, choices=['validation','test'])
    metric.add_argument('--precision', type=float); metric.add_argument('--recall', type=float)
    metric.add_argument('--map50', type=float); metric.add_argument('--map5095', type=float)
    metric.add_argument('--fps', type=float); metric.add_argument('--evidence', required=True)
    ev=sub.add_parser('evaluate',help='Evaluate a finished custom training run on video-held-out validation')
    ev.add_argument('--run',type=Path,help='Training run directory (defaults to newest)')
    ev.add_argument('--data',type=Path)
    ev.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    train = sub.add_parser('train', help='Run your own real-image scratch detector; update report afterward')
    train.add_argument('--mode', required=True, choices=['quick','full'])
    train.add_argument('--data', type=Path)
    train.add_argument('--device', default='auto', choices=['auto','cpu','cuda'])
    train.add_argument('--epochs', type=int)
    args = p.parse_args()
    initialize()
    if args.command == 'note':
        rows = read_csv(RECORDS / 'milestones.csv')
        rows.append(dict(zip(EVENT_FIELDS, [datetime.now().date().isoformat(), args.status, args.title, args.detail])))
        write_csv(RECORDS / 'milestones.csv', EVENT_FIELDS, rows)
    elif args.command == 'metrics':
        for name in ('precision','recall','map50','map5095'):
            val = getattr(args,name)
            if val is not None and not (0 <= val <= 1):
                p.error(f'--{name} must be a decimal fraction from 0 to 1 (e.g. 0.75 = 75%).')
        if args.fps is not None and args.fps <= 0: p.error('--fps must be positive')
        if all(getattr(args,k) is None for k in ('precision','recall','map50','map5095','fps')):
            p.error('Provide at least one measured metric.')
        rows = read_csv(RECORDS / 'model_metrics.csv')
        row = {'timestamp':datetime.now().isoformat(timespec='seconds'),
               'model':args.model,'training':args.training,'split':args.split,
               'evidence':args.evidence}
        for key in ('precision','recall','map50','map5095','fps'):
            row[key] = getattr(args,key) if getattr(args,key) is not None else ''
        rows.append(row); write_csv(RECORDS / 'model_metrics.csv', FIELDS, rows)
    elif args.command == 'update':
        if args.metrics_json:
            was_new=register_evaluation_json(args.metrics_json.resolve(),args.mode)
            print('Evaluation metrics imported' if was_new else 'Evaluation already recorded')
        if args.loss_csv:
            is_new = register_loss_csv(args.loss_csv.resolve(), args.mode)
            print('New run imported' if is_new else 'Existing run already recorded')
        else:
            for path in sorted(RUNS.glob('*/loss_history.csv')):
                register_loss_csv(path, mode=('quick' if '_quick' in path.parent.name else 'full'))
    elif args.command == 'evaluate':
        data=get_data_root(args.data)
        if data is None:p.error('Cannot locate prepared BIRDSAI; use --data "FULL_PATH_TO_yolo_real"')
        run_dir=args.run if args.run else next(iter(sorted([x for x in RUNS.iterdir() if x.is_dir()],reverse=True)),None) if RUNS.exists() else None
        if run_dir is None:p.error('No training run has completed. Run train --mode quick first.')
        ck=Path(run_dir)/'thermal_scout_real_weights.pt'
        if not ck.is_file():p.error(f'Checkpoint missing: {ck}')
        output=Path(run_dir)/'evaluation.json'
        cmd=[sys.executable,'-m', 'thermalscout.evaluate_detector',
             '--checkpoint',str(ck),'--data',str(data),'--out',str(output),'--device',args.device]
        print('Starting evaluation of DEVELOPMENT validation videos...',flush=True)
        result=subprocess.run(cmd, cwd=ROOT)
        if result.returncode:raise SystemExit(result.returncode)
        mode='quick' if '_quick' in Path(run_dir).name else 'full'
        register_evaluation_json(output,mode)
    elif args.command == 'train':
        data = get_data_root(args.data)
        if data is None:
            p.error('Prepared images not found. Pass --data "FULL_PATH_TO_yolo_real".')
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out = RUNS / f'{stamp}_{args.mode}'
        cmd = [sys.executable, '-m', 'thermalscout.train_detector',
               f'--{args.mode}', '--data', str(data), '--out', str(out), '--device',args.device]
        if args.epochs: cmd += ['--epochs',str(args.epochs)]
        print('Starting custom detector training:', ' '.join(cmd), flush=True)
        result = subprocess.run(cmd)
        if result.returncode:
            print('Training did not complete. Check the error above; no results were invented.')
            raise SystemExit(result.returncode)
        register_loss_csv(out / 'loss_history.csv', args.mode)
        print('Recorded training source:', out / 'loss_history.csv')
    report_update()

if __name__ == '__main__':
    main()
