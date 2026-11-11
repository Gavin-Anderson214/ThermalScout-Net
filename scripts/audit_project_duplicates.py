"""Read-only inventory of identical small source files in historical project folders."""
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    parent=root.parent.parent
    roots=[root,parent/'ThermalScout_BIRDSAI_Step2',parent/'ThermalScout_Demo']
    groups=defaultdict(list)
    skip={'data','training_runs','.git','.venv','.venv-1','__pycache__','node_modules','TrainReal','TestReal','dist','birdsai_step2_output','yolo_real','images','labels'}
    for base in roots:
        for folder,dirs,files in os.walk(base):
            dirs[:]=[d for d in dirs if d not in skip]
            for name in files:
                file=Path(folder)/name
                if file.suffix not in {'.py','.md','.txt','.yaml','.yml'} or file.stat().st_size>1000000: continue
                groups[hashlib.sha256(file.read_bytes()).hexdigest()].append(str(file.relative_to(parent)))
    duplicates=[{'sha256':h,'paths':paths} for h,paths in groups.items() if len(paths)>1]
    report=dict(scope='Small source/doc files only; excludes data, checkpoints, environments and Git',
                duplicate_groups=duplicates,removed=[],decision='Preserve historical copies: byte equality alone does not establish that imports, relative paths or prior experiments no longer need them.')
    out=root/'records/experiments/20261110_bambi_repair/duplicate_audit.json'
    out.write_text(json.dumps(report,indent=2))
    print(f'{len(duplicates)} identical-file groups. No files deleted or moved.')

if __name__=='__main__':main()
