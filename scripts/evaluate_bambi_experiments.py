"""Save reproducible threshold and per-flight diagnostics for fixed checkpoints."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from thermalscout.model import ThermalScoutReal
from thermalscout.data import BirdsaiDataset
from thermalscout.inference import infer, create_preview
from thermalscout.evaluate_detector import get_counts, ap_101

def summarize(entries):
    aps=[ap_101(entries,c,.5) for c in (0,1)]
    return dict(map50=float(np.mean([a for a in aps if a is not None])), ap50=aps,
                counts={str(t):{name:get_counts(entries,c,conf=t) for c,name in enumerate(('animal','human'))}
                        for t in (.25,.5,.6,.7,.8,.9)})

def main():
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1]
    out=root/'records/experiments/20261110_bambi_repair'
    result={}
    for name in ('control','hardneg','hardneg_bg1'):
        run=root/f'training_runs/bambi_{name}_40ep_20261110'
        path=run/'thermal_scout_real_weights.pt'
        ck=torch.load(path,map_location='cuda',weights_only=True)
        model=ThermalScoutReal().cuda();model.load_state_dict(ck['model']);model.eval()
        result[name]=dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),epoch=ck['epoch'])
        for split in ('train','val'):
            ds=BirdsaiDataset(root/'data/bambi_pilot',split,512,0,42)
            for peak in (False,True) if split=='val' else (False,):
                entries=[]
                for image,boxes,path in ds:
                    entries.append(dict(path=Path(path).name,ground_truth=boxes,predictions=infer(model,image,'cuda',.001,local_peaks=peak)))
                key=split+('_peaks' if peak else '')
                result[name][key]=summarize(entries)
                if split=='val':
                    result[name][key]['flights']={str(f):summarize([e for e in entries if e['path'].startswith(f'flight{f}_')]) for f in (6,22)}
        create_preview(model,ds,'cuda',run/'preview_conf050.png',.5)
        # Archive small experiment records without checkpoint or imagery.
        for file in ('config.json','loss_history.csv'):
            (out/f'{name}_{file}').write_bytes((run/file).read_bytes())
        print(name,result[name]['val']['map50'],result[name]['val_peaks']['map50'],flush=True)
    (out/'comparison.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
