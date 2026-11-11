"""Verify pilot labels against released MOT rows, flight isolation and file hashes."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from thermalscout.data import read_boxes

ANIMALS = {'Sus scrofa', 'Cervus elaphus', 'Capreolus capreolus', 'Dama dama',
           'Capra ibex', 'Rupicapra rupicapra', 'Aves', 'Canis lupus familiaris',
           'Sus scrofa × Sus domesticus'}

def species_class(name):
    species = name.split(' (')[0].strip()
    if species == 'Homo sapiens': return 1
    if species in ANIMALS: return 0
    return None  # Unknown and No-animal must not silently become animal labels.

def main():
    import cv2  # Optional video dependency; species tests need no OpenCV.
    root=Path(__file__).resolve().parents[1]
    source=root/'data/bambi_source/bambi_downloads'
    result={'source':'https://github.com/bambi-eco/Dataset/blob/main/docs/annotation-format.md',
            'splits':{}, 'issues':[], 'manifest':[]}
    hashes=defaultdict(list)
    all_flights={}
    for split in ('train','val'):
        counts=Counter(); species=Counter(); flights=set(); frame_count=0
        for image in sorted((root/'data/bambi_pilot/images'/split).glob('*.png')):
            frame_count+=1
            flight=int(image.stem.split('_')[0][6:]); frame=int(image.stem.split('frame')[1])
            flights.add(flight)
            rows=list(csv.reader((source/f'{flight}_gt.txt').open(encoding='utf-8')))
            selected=[r for r in rows if int(r[0])==frame]
            expected=[]
            for row in selected:
                c=species_class(row[9]); species[row[9]]+=1
                if c is None:
                    result['issues'].append(f'{image.name}: excluded/unmapped species {row[9]}')
                    continue
                x,y,w,h=map(float,row[2:6])
                if w<=0 or h<=0 or min(x,y)<0 or x+w>1024 or y+h>1024:
                    result['issues'].append(f'{image.name}: invalid source box'); continue
                expected.append((c,(x+w/2)/1024,(y+h/2)/1024,w/1024,h/1024))
            label=root/'data/bambi_pilot/labels'/split/(image.stem+'.txt')
            actual=read_boxes(label)
            if len(actual)!=len(expected) or not np.allclose(sorted(actual),sorted(expected),atol=5.1e-7,rtol=0):
                result['issues'].append(f'{image.name}: label mismatch')
            counts.update(int(b[0]) for b in actual)
            digest=hashlib.sha256(image.read_bytes()).hexdigest(); hashes[digest].append(f'{split}/{image.name}')
            result['manifest'].append(dict(split=split,image=image.name,flight=flight,frame=frame,sha256=digest,label_sha256=hashlib.sha256(label.read_bytes()).hexdigest()))
        all_flights[split]=flights
        result['splits'][split]=dict(frames=frame_count,flights=sorted(flights),labels=dict(counts),species=dict(species))
    result['flight_overlap']=sorted(all_flights['train'] & all_flights['val'])
    result['duplicate_images']=[v for v in hashes.values() if len(v)>1]
    # Verify extraction against actual video bytes at the named zero-based index.
    result['video_checks']=[]
    for flight in (47,6,22):
        item=next(m for m in result['manifest'] if m['flight']==flight)
        cap=cv2.VideoCapture(str(source/f'{flight}_matched_processed.mp4'))
        cap.set(cv2.CAP_PROP_POS_FRAMES,item['frame']); ok,frame=cap.read();cap.release()
        if not ok: raise RuntimeError('Video decode failed')
        expected=cv2.resize(cv2.cvtColor(frame[:,:1024],cv2.COLOR_BGR2GRAY),(512,512),interpolation=cv2.INTER_AREA)
        actual=cv2.imread(str(root/'data/bambi_pilot/images'/item['split']/item['image']),0)
        result['video_checks'].append(dict(flight=flight,frame=item['frame'],shape=list(frame.shape),max_pixel_error=int(np.abs(expected.astype(int)-actual.astype(int)).max())))
    target=root/'records/experiments/20261110_bambi_repair/dataset_audit.json'
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='manifest'},indent=2))
    if result['issues'] or result['flight_overlap']: raise SystemExit('Audit needs review')

if __name__=='__main__': main()
