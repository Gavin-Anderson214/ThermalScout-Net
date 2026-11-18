"""Reevaluate named candidates on one development split and retain measured best.

Example: --candidate baseline=training_runs/baseline --candidate variant=training_runs/variant
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from thermalscout.evaluate_detector import get_counts
from thermalscout.model import CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', action='append', required=True, help='Label=run_directory')
    parser.add_argument('--data', type=Path, default=Path('data/processed/yolo_real'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    comparison = []
    for index, item in enumerate(args.candidate):
        label, directory = item.split('=', 1)
        run = Path(directory)
        checkpoint = run/'thermal_scout_real_weights.pt'
        output = args.out/f'candidate_{index}_evaluation.json'
        predictions = args.out/f'candidate_{index}_predictions.json'
        command = [sys.executable, '-u', '-c',
                   "import torch,runpy; torch.set_num_threads(4); runpy.run_module('thermalscout.evaluate_detector',run_name='__main__')",
                   '--checkpoint', str(checkpoint), '--data', str(args.data), '--out', str(output),
                   '--predictions', str(predictions), '--device', 'cuda']
        subprocess.run(command, check=True)
        measured = json.loads(output.read_text())
        comparison.append(dict(label=label, run=run.as_posix(), index=index, **measured))
    best = max(comparison, key=lambda r:(r['map50'],r['class_metrics']['human']['ap50']))
    shutil.copy2(Path(best['run'])/'thermal_scout_real_weights.pt', args.out/'thermal_scout_real_weights.pt')
    shutil.copy2(args.out/f"candidate_{best['index']}_predictions.json", args.out/'predictions.json')
    thresholds = []
    entries = json.loads((args.out/'predictions.json').read_text())
    for threshold in [.25,.35,.5,.65,.75,.85,.9,.95,.975,.99,.995,.999]:
        for c,name in enumerate(CLASSES):
            count = get_counts(entries,c,conf=threshold)
            count['f1'] = 2*count['precision']*count['recall']/max(1e-12,count['precision']+count['recall'])
            thresholds.append(dict(threshold=threshold, class_name=name, **count))
    with (args.out/'thresholds.csv').open('w', newline='', encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(thresholds[0]))
        writer.writeheader(); writer.writerows(thresholds)
    selection = dict(criterion='Highest development mAP50 across named candidates; human AP50 breaks ties',
                     selected_run=best['run'], evaluation={k:v for k,v in best.items() if k not in ('label','run','index')},
                     comparison=comparison,
                     threshold_scope='Exploratory operating points on the same development set. Optimistic after selection; no independent test.',
                     best_development_f1_by_class={name:max((r for r in thresholds if r['class_name']==name),key=lambda r:r['f1']) for name in CLASSES})
    (args.out/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print(json.dumps(selection,indent=2),flush=True)


if __name__=='__main__':
    main()
