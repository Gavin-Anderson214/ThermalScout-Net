"""Draw the same human-heavy validation frames for both class-weight models."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from thermalscout.data import BirdsaiDataset
from thermalscout.model import ThermalScoutReal
from thermalscout.inference import infer
from thermalscout.evaluate_detector import xywh_to_xyxy, get_counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gains',action='store_true',help='Select the largest per-frame human TP gains, one per video')
    args=parser.parse_args()
    ds=BirdsaiDataset(Path('data/processed/yolo_real'),'val',256,0,42,augment=False)
    models=[]
    for tag in ('control','human4'):
        ck=torch.load(f'training_runs/20261102_classweight_{tag}/thermal_scout_real_weights.pt',map_location='cpu',weights_only=True)
        model=ThermalScoutReal().eval();model.load_state_dict(ck['model']);models.append(model)
    chosen=sorted(range(len(ds)),key=lambda i:-sum(int(b[0])==1 for b in ds.records[i][2]))[:2]
    if args.gains:
        old=json.loads(Path('training_runs/20261029_133235_full/error_analysis/predictions.json').read_text())
        new=json.loads(Path('training_runs/20261102_classweight_human4_analysis/predictions.json').read_text())
        assert len(old)==len(new)==len(ds)
        gains=[]
        for i,(a,b) in enumerate(zip(old,new)):
            assert a['image']==b['image']==ds.records[i][0].name
            gain=get_counts([b],1)['tp']-get_counts([a],1)['tp']
            if gain>0:gains.append((gain,i))
        chosen=[];videos=set()
        for gain,i in sorted(gains,key=lambda t:(-t[0],t[1])):
            video=ds.records[i][0].stem.split('__f')[0]
            if video in videos:continue
            chosen.append(i);videos.add(video)
            if len(chosen)==2:break
        if not chosen:raise ValueError('No frames with human TP gains')
        print('Deliberately selected largest human TP gain per video:',[ds.records[i][0].name for i in chosen])
    side,header=512,44
    canvas=Image.new('RGB',(3*side,len(chosen)*(side+header)),'#152033');pen=ImageDraw.Draw(canvas)
    for row,index in enumerate(chosen):
        tensor,boxes,path=ds[index]
        panels=[('Labels',[(int(b[0]),1,xywh_to_xyxy(b)[1]) for b in boxes])]
        panels += [(f'Human weight {weight} | confidence .25',infer(model,tensor,'cpu',threshold=.001)) for weight,model in zip((2,4),models)]
        for col,(title,dets) in enumerate(panels):
            image=Image.fromarray((tensor[0].numpy()*255).astype('uint8')).convert('RGB').resize((side,side));draw=ImageDraw.Draw(image)
            for c,score,box in dets:
                if score<.25:continue
                coords=[v*side for v in box];color='#53c6ff' if c else '#a8ff78'
                draw.rectangle(coords,outline=color,width=2)
                draw.text((coords[0],max(0,coords[1]-12)),f'{"human" if c else "animal"} {score:.2f}',fill=color)
            x,y=col*side,row*(side+header)
            pen.text((x+5,y+3),title,fill='white');pen.text((x+5,y+22),Path(path).stem,fill='white');canvas.paste(image,(x,y+header))
    out=Path('training_runs/20261102_classweight_gains.png' if args.gains else 'training_runs/20261102_classweight_comparison.png');canvas.save(out);print(out.resolve())


if __name__=='__main__':main()
