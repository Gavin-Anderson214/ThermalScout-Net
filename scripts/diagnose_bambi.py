"""Reproduce pilot outputs and isolate BatchNorm statistics using train data only."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from thermalscout.data import BirdsaiDataset, collate
from thermalscout.model import ThermalScoutReal
from thermalscout.inference import infer
from thermalscout.evaluate_detector import get_counts, ap_101

def evaluate(model, ds):
    entries = []
    for image, boxes, path in ds:
        entries.append(dict(path=Path(path).name, ground_truth=boxes,
                            predictions=infer(model, image, 'cuda', .001)))
    metrics = {name: dict(get_counts(entries, c), ap50=ap_101(entries, c, .5))
               for c, name in enumerate(('animal', 'human'))}
    return dict(classes=metrics, map50=sum(m['ap50'] for m in metrics.values())/2)

def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    out = root / 'records/experiments/20261110_bambi_repair'
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = root / 'training_runs/bambi_backup_gpu_pilot_v2/thermal_scout_real_weights.pt'
    ck = torch.load(checkpoint, weights_only=True, map_location='cuda')
    model = ThermalScoutReal().cuda(); model.load_state_dict(ck['model']); model.eval()
    train = BirdsaiDataset(root/'data/bambi_pilot', 'train', 512, 0, 42)
    val = BirdsaiDataset(root/'data/bambi_pilot', 'val', 512, 0, 42)
    report = dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  parameters=sum(p.numel() for p in model.parameters()), device=torch.cuda.get_device_name(),
                  train_counts=train.label_counts, val_counts=val.label_counts)
    with torch.no_grad():
        images = torch.stack([val[i][0] for i in range(5)]).cuda()
        pred = model(images)
        report['input_std'] = images.flatten(1).std(1).tolist()
        report['output_mean_absolute_difference'] = [(pred[0]-p).abs().mean().item() for p in pred[1:]]
        report['objectness_range'] = [pred[:,0].sigmoid().min().item(), pred[:,0].sigmoid().max().item()]
    report['original'] = evaluate(model,val)
    print('Original', report, flush=True)
    calibrated = copy.deepcopy(model)
    # Separate saved copy: reset running moments then average training batches.
    for module in calibrated.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.reset_running_stats(); module.momentum=None
    calibrated.train()
    with torch.no_grad():
        for images, _, _ in DataLoader(train,batch_size=2,collate_fn=collate):
            calibrated(images.cuda())
    calibrated.eval()
    report['train_only_bn_calibration'] = evaluate(calibrated,val)
    with torch.no_grad():
        pred=calibrated(torch.stack([val[i][0] for i in range(5)]).cuda())
        report['calibrated_output_difference']=[(pred[0]-p).abs().mean().item() for p in pred[1:]]
    (out/'diagnosis.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
