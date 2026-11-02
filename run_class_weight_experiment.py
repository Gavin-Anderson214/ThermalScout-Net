"""Run a seed-42, ten-epoch baseline-versus-human-weight experiment."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
from thermalscout.data import BirdsaiDataset


def main():
    root=Path('data/processed/yolo_real')
    output=Path('records/experiments/20261102_human_weight')
    output.mkdir(parents=True,exist_ok=True)
    audit={}; digest=hashlib.sha256()
    for split in ('train','val'):
        ds=BirdsaiDataset(root,split,256,0,42,augment=False)
        sizes=[[],[]]; videos=set()
        for image,label,boxes in ds.records:
            digest.update(str(image.relative_to(root)).encode());digest.update(image.read_bytes());digest.update(label.read_bytes())
            videos.add(image.stem.split('__f')[0])
            with Image.open(image) as im: w,h=im.size
            scale=min(256/w,256/h);nw,nh=round(w*scale),round(h*scale)
            for c,x,y,bw,bh in boxes: sizes[int(c)].append(min(bw*nw,bh*nh))
        audit[split]=dict(images=len(ds),labels=ds.label_counts,videos=sorted(videos),
            min_box_side_pixels={name:dict(p10=float(np.percentile(sizes[c],10)),median=float(np.median(sizes[c])),
                                          under4=int(sum(v<4 for v in sizes[c]))) for c,name in enumerate(('animal','human'))})
    assert not set(audit['train']['videos']) & set(audit['val']['videos'])
    audit['dataset_sha256']=digest.hexdigest()
    audit['class_mapping']='Prepared data.yaml and loader agree: 0 animal, 1 human; original annotations not independently remapped.'
    (output/'dataset_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2),flush=True)
    configs=[]
    for weight,tag in [(2,'control'),(4,'human4')]:
        run=Path(f'training_runs/20261102_classweight_{tag}')
        cmd=[sys.executable,'-u','-m','thermalscout.train_detector','--full','--data',str(root),'--out',str(run),
             '--epochs','10','--size','256','--batch','4','--lr','0.001','--seed','42',
             '--device','cpu','--objectness','baseline','--human-weight',str(weight)]
        log=output/f'{tag}_training.log'
        with log.open('w',encoding='utf-8') as fp:
            proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
            for line in proc.stdout:
                print(line,end='',flush=True);fp.write(line);fp.flush()
            if proc.wait(): raise RuntimeError(f'{tag} training failed; see {log}')
        subprocess.run([sys.executable,'-u','-m','thermalscout.evaluate_detector','--checkpoint',str(run/'thermal_scout_real_weights.pt'),
                        '--data',str(root),'--out',str(run/'evaluation.json'),'--device','cpu'],check=True)
        for name in ['config.json','loss_history.csv','evaluation.json']:
            (output/f'{tag}_{name}').write_bytes((run/name).read_bytes())
        configs.append(json.loads((run/'config.json').read_text()))
    ignored={'out','command','human_weight'}
    assert {k:v for k,v in configs[0].items() if k not in ignored}=={k:v for k,v in configs[1].items() if k not in ignored}
    print('Both experiments completed; configurations differ only by human class weight and output identity.',flush=True)


if __name__=='__main__': main()
