"""Draw measured TP/FP/FN examples from an evaluator's local prediction cache."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from thermalscout.evaluate_detector import xywh_to_xyxy, iou_xyxy, get_counts
from thermalscout.model import CLASSES


def classify(entry):
    truth = [xywh_to_xyxy(b) for b in entry['ground_truth']]
    used = set()
    predictions = []
    for c, score, box in sorted(entry['predictions'], key=lambda p:-p[1]):
        if score < .25:
            continue
        overlap,j = max(((iou_xyxy(box,t),j) for j,(tc,t) in enumerate(truth)
                         if tc==c and j not in used), default=(0,-1))
        hit = j>=0 and overlap>=.5
        if hit:
            used.add(j)
        predictions.append((c, score, box, 'TP' if hit else 'FP'))
    missed = [(c,box) for j,(c,box) in enumerate(truth) if j not in used]
    return predictions, missed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--size',type=int,default=512)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    entries=json.loads(args.predictions.read_text())
    counts=[{name:get_counts([e],c) for c,name in enumerate(CLASSES)} for e in entries]
    candidates=[
        ('correct_animal',max(range(len(entries)),key=lambda i:counts[i]['animal']['tp']),
         'Most matched animal boxes at confidence 0.25'),
        ('correct_human',max(range(len(entries)),key=lambda i:counts[i]['human']['tp']),
         'Most matched human boxes at confidence 0.25'),
        ('false_positives',max(range(len(entries)),key=lambda i:sum(v['fp'] for v in counts[i].values())),
         'Most false positive boxes at confidence 0.25'),
        ('missed_humans',max(range(len(entries)),key=lambda i:counts[i]['human']['fn']),
         'Most missed human boxes at confidence 0.25')]
    manifest=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    for name,i,reason in candidates:
        entry=entries[i]
        with Image.open(entry['path']) as im:
            im=im.convert('L')
            width,height=im.size
            scale=min(args.size/width,args.size/height)
            rw,rh=round(width*scale),round(height*scale)
            image=Image.new('RGB',(args.size,args.size),'black')
            image.paste(im.resize((rw,rh)),((args.size-rw)//2,(args.size-rh)//2))
        # Draw directly in the same letterboxed coordinates used by evaluation.
        image=image.resize((1024,1024))
        slide_image=image.copy()
        draw=ImageDraw.Draw(image)
        predictions,missed=classify(entry)
        for c,score,box,status in predictions:
            pixels=[v*1024 for v in box]
            color='#40ff80' if status=='TP' else '#ff6575'
            draw.rectangle(pixels,outline=color,width=2)
            label=f'{status} {CLASSES[c]} {score:.3f}'
            x,y=pixels[0],max(0,pixels[1]-18)
            bounds=draw.textbbox((x,y),label,font=font)
            draw.rectangle(bounds,fill='black')
            draw.text((x,y),label,font=font,fill=color)
        for c,box in missed:
            pixels=[v*1024 for v in box]
            draw.rectangle(pixels,outline='#55c8ff',width=3)
            draw.text((pixels[0],min(1000,pixels[3]+2)),f'MISS {CLASSES[c]}',font=font,fill='#55c8ff',stroke_width=1,stroke_fill='black')
        image.save(args.out/f'{name}.png')
        # Number a few representative boxes for readable native slide captions.
        # Keep every predicted and missed box visible in this overview.
        overview=ImageDraw.Draw(slide_image)
        for c,score,box,status in predictions:
            overview.rectangle([v*1024 for v in box],outline='#40ff80' if status=='TP' else '#ff6575',width=2)
        for c,box in missed:
            overview.rectangle([v*1024 for v in box],outline='#55c8ff',width=3)
        correct=[p for p in predictions if p[3]=='TP'][:2]
        false=[p for p in predictions if p[3]=='FP'][:(1 if correct else 2)]
        highlights=[]
        for c,score,box,status in correct+false:
            highlights.append(dict(number=len(highlights)+1,status=status,class_name=CLASSES[c],confidence=score,box=box))
        for c,box in missed[:max(0,4-len(highlights))]:
            highlights.append(dict(number=len(highlights)+1,status='MISS',class_name=CLASSES[c],confidence=None,box=box))
        number_font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',32)
        positions=[]
        for item in highlights:
            x=max(0,min(970,item['box'][0]*1024)); y=max(0,min(970,item['box'][1]*1024-36))
            anchor=(x,item['box'][1]*1024)
            while any(abs(x-px)<48 and abs(y-py)<42 for px,py in positions):
                if y+44<970:
                    y+=44
                else:
                    x=max(0,x-48)
            positions.append((x,y))
            label=str(item['number'])
            color={'TP':'#40ff80','FP':'#ff6575','MISS':'#55c8ff'}[item['status']]
            overview.line((anchor,(x+10,y+18)),fill=color,width=2)
            overview.rectangle(overview.textbbox((x,y),label,font=number_font),fill='black')
            overview.text((x,y),label,font=number_font,fill=color)
        slide_image.save(args.out/f'{name}_slide.png')
        manifest.append(dict(name=name,image=Path(entry['path']).name,selection=reason,
                             counts=counts[i],highlights=highlights,predictions=predictions,missed=missed))
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in m.items() if k not in ('predictions','missed')} for m in manifest],indent=2))


if __name__=='__main__':
    main()
