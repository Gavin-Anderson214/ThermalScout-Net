"""Launch ThermalScout training with a local browser graph and live console.

Run beside project_manager.py: python3.13 live_dashboard.py --mode full
Only Python's standard library is needed for the dashboard itself.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import webbrowser

ROOT = Path(__file__).resolve().parent
EPOCH = re.compile(r"Epoch\s+(\d+)/(continuous|\d+): train loss=([\d.eE+-]+), val loss=([\d.eE+-]+)")

# The page draws SVG lines from real epoch measurements; nothing is simulated.
PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ThermalScout · Training mission control</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#0b111b;color:#e9eef6;font:16px system-ui,sans-serif}
main{max-width:1200px;margin:auto;padding:36px 24px}small,.muted{color:#9aacc2}
header{display:flex;justify-content:space-between;align-items:center;gap:20px}h1{font-size:32px;margin:8px 0}
.tag{color:#90f5c0;letter-spacing:.13em;font-size:12px}#status{padding:9px 15px;border:1px solid #36506a;border-radius:24px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0 18px}
.card,.panel{background:#131e2e;border:1px solid #24354c;border-radius:16px;padding:20px}.value{font-size:29px;margin:8px 0}
.bar{height:7px;background:#23344a;border-radius:9px;overflow:hidden;margin:15px 0}.bar span{display:block;height:100%;background:#88efb5;transition:width .5s}
h2{font-size:18px;margin:0 0 10px}.row{display:flex;gap:20px;flex-wrap:wrap;margin:14px 0}.train{color:#8bf3b8}.val{color:#68b9ff}
svg{display:block;width:100%;height:auto}svg text{font:13px system-ui;fill:#9aacc2}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:310px;overflow:auto;color:#c6d4e7;font:13px/1.7 Consolas,monospace}
.panel{margin-bottom:18px}#preview{max-width:100%;border-radius:8px}.error{color:#ff9e9e}button{background:#23344a;color:white;border:1px solid #45607e;border-radius:8px;padding:8px;cursor:pointer}
@media(max-width:700px){.cards{grid-template-columns:repeat(2,1fr)}header{align-items:flex-start;flex-direction:column}h1{font-size:26px}}
</style><main>
<header><div><div class="tag">THERMALSCOUT-NET / LIVE TRAINING</div><h1>Training mission control</h1><small>Real BIRDSAI images · Custom CNN · Development validation</small></div><div id="status">Connecting…</div></header>
<section class="cards"><div class="card"><small>Completed epochs</small><div class="value" id="epoch">—</div></div><div class="card"><small>Latest training loss</small><div class="value train" id="train">—</div></div><div class="card"><small>Best validation loss</small><div class="value val" id="best">—</div></div><div class="card"><small>Elapsed time</small><div class="value" id="elapsed">—</div></div></section>
<section class="panel"><h2 id="stage">Preparing training…</h2><div class="bar"><span id="progress" style="width:0%"></span></div><small id="details">Waiting for model and dataset details.</small></section>
<section class="panel"><h2>Learning curve</h2><div class="row"><span class="train">● Training loss</span><span class="val">● Validation loss</span></div><svg id="chart" viewBox="0 0 1000 350" role="img" aria-label="Training and validation loss by completed epoch"></svg><small id="chartHelp">Lower loss means fewer errors under the training objective. It is not an accuracy percentage. Points appear after each epoch, not after each batch.</small></section>
<section class="panel" id="imagePanel" hidden><h2>Labels and predictions</h2><img id="preview" alt="Validation labels beside model predictions"><p class="muted">Generated from the best checkpoint at the end of training. This single image does not measure overall accuracy.</p></section>
<section class="panel"><h2>Live activity</h2><label><input id="follow" type="checkbox" checked> Follow new output</label> <button id="download">Save console log</button><pre id="log"></pre></section>
<small id="footer">Local dashboard · Keep PowerShell open · Ctrl+C in PowerShell stops the dashboard and its training process. Do not start a second training run alongside one already running.</small>
</main><script>
let latest=null;
function draw(points,total){
 const svg=document.getElementById('chart'); svg.replaceChildren();
 const ns='http://www.w3.org/2000/svg';
 function el(name,attrs,text){const e=document.createElementNS(ns,name);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;svg.appendChild(e);return e}
 const left=65,right=975,top=20,bottom=295;
 const max=Math.max(.1,...points.flatMap(p=>[p.train,p.val]))*1.12;
 for(let i=0;i<=4;i++){let y=bottom-i*(bottom-top)/4;el('line',{x1:left,y1:y,x2:right,y2:y,stroke:'#293a50'});el('text',{x:left-10,y:y+4,'text-anchor':'end'},(max*i/4).toFixed(2))}
 const x=n=>left+(n-1)/Math.max(1,total-1)*(right-left), y=v=>bottom-v/max*(bottom-top);
 el('text',{x:520,y:343,'text-anchor':'middle'},'Completed epoch');
 const tickStep=Math.max(1,Math.ceil(total/10));
 for(let i=1;i<=total;i+=tickStep)el('text',{x:x(i),y:318,'text-anchor':'middle'},i);
 if((total-1)%tickStep!==0)el('text',{x:x(total),y:318,'text-anchor':'middle'},total);
 for(const [key,color]of [['train','#8bf3b8'],['val','#68b9ff']]){
 el('polyline',{points:points.map(p=>`${x(p.epoch)},${y(p[key])}`).join(' '),fill:'none',stroke:color,'stroke-width':3});
 for(const p of points){const dot=el('circle',{cx:x(p.epoch),cy:y(p[key]),r:4,fill:color});const title=document.createElementNS(ns,'title');title.textContent=`Epoch ${p.epoch}: ${key} loss ${p[key]}`;dot.appendChild(title)} }
 if(!points.length)el('text',{x:520,y:155,'text-anchor':'middle'},'Waiting for the first completed epoch…');
}
async function poll(){try{
 const r=await fetch('/state',{cache:'no-store'});if(!r.ok)throw Error('Connection failed');const s=await r.json();latest=s;
 document.getElementById('status').textContent=s.status;
 document.getElementById('stage').textContent=s.stage;
 document.getElementById('details').textContent=s.details||'Loading libraries and dataset…';
 if(s.watching){
 document.getElementById('footer').textContent='Local dashboard · Watching the files from an existing run. Its PowerShell window controls training.';
 document.getElementById('chartHelp').textContent='This chart uses the epoch measurements saved in the run folder at the end of training.';
 }
 const continuous=s.total==='continuous';
 const recorded=s.watching&&!s.points.length?'—':s.points.length;
 document.getElementById('epoch').textContent=continuous?`${recorded} completed`:`${recorded} / ${s.total}`;
 const last=s.points.at(-1);document.getElementById('train').textContent=last?last.train.toFixed(4):'—';
 document.getElementById('best').textContent=last?Math.min(...s.points.map(p=>p.val)).toFixed(4):'—';
 const secs=Math.floor(s.elapsed);document.getElementById('elapsed').textContent=`${Math.floor(secs/60)}m ${secs%60}s`;
 document.getElementById('progress').style.width=continuous?'100%':`${100*s.points.length/s.total}%`;
 const log=document.getElementById('log');log.textContent=s.lines.join('\n');if(document.getElementById('follow').checked)log.scrollTop=log.scrollHeight;
 draw(s.points,continuous?Math.max(10,last?.epoch||1):s.total);if(s.preview){document.getElementById('imagePanel').hidden=false;document.getElementById('preview').src='/preview'}
 }catch(e){document.getElementById('status').textContent='Disconnected — check PowerShell'}finally{setTimeout(poll,1200)}}
document.getElementById('download').onclick=()=>{if(!latest)return;const a=document.createElement('a');const url=URL.createObjectURL(new Blob([latest.lines.join('\n')],{type:'text/plain'}));a.href=url;a.download='thermalscout-console.txt';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};poll();
</script></html>'''


class Monitor:
    """Keep measurements and console output together under a thread lock."""
    def __init__(self, total):
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.finished = None
        self.total = total
        self.points = []
        self.lines = deque(maxlen=5000)
        self.status = 'Starting'
        self.stage = 'Loading Python libraries and prepared images'
        self.details = ''
        self.preview = None

    def record(self, line):
        with self.lock:
            self.lines.append(line)
            if line.startswith(('Device:', 'Mode:')):
                self.details += (' · ' if self.details else '') + line
            if line.startswith('Device:'):
                self.status = 'Training'
                self.stage = 'Training first epoch — waiting for measurements'
            match = EPOCH.search(line)
            if match:
                epoch, total, train, val = match.groups()
                self.total = 'continuous' if total == 'continuous' else int(total)
                self.points.append(dict(epoch=int(epoch), train=float(train), val=float(val)))
                self.stage = (f'Training epoch {int(epoch) + 1} of continuous'
                              if total == 'continuous' else
                              (f'Training epoch {int(epoch) + 1} of {total}' if int(epoch) < int(total)
                               else 'Saving best checkpoint, prediction preview and report'))
            if line.startswith('Preview: '):
                candidate = Path(line[9:].split(' (', 1)[0]).resolve()
                # Only serve a generated PNG from this project's training outputs.
                if candidate.is_relative_to((ROOT / 'training_runs').resolve()) and candidate.suffix == '.png':
                    self.preview = candidate

    def snapshot(self):
        with self.lock:
            return dict(total=self.total, points=list(self.points), lines=list(self.lines),
                        status=self.status, stage=self.stage, details=self.details,
                        elapsed=(self.finished or time.monotonic())-self.started,
                        preview=bool(self.preview and self.preview.is_file()))


class SavedRunMonitor:
    """Show evidence from a run started in another PowerShell window."""
    def __init__(self, run):
        self.run = run
        config = run / 'config.json'
        settings = json.loads(config.read_text(encoding='utf-8'))
        self.total = 'continuous' if settings.get('continuous') else int(settings['epochs'])
        self.started = config.stat().st_mtime
        self.preview = run / 'real_detection_preview.png'

    def snapshot(self):
        points = []
        history = self.run / 'loss_history.csv'
        if history.is_file():
            try:
                with history.open(encoding='utf-8', newline='') as fp:
                    for row in csv.DictReader(fp):
                        points.append(dict(epoch=int(row['epoch']), train=float(row['train_loss']),
                                           val=float(row['val_loss'])))
            except (OSError, KeyError, TypeError, ValueError):
                # The trainer may still be writing the final history.
                points = []
        complete = self.total != 'continuous' and len(points) == self.total and self.preview.is_file()
        checkpoint = self.run / 'thermal_scout_real_weights.pt'
        lines = [f'Run: {self.run.name}']
        if checkpoint.is_file():
            updated = datetime.fromtimestamp(checkpoint.stat().st_mtime).strftime('%H:%M:%S')
            lines.append(f'Best checkpoint last saved at {updated}.')
        if points:
            lines.extend(f"Epoch {point['epoch']}: train loss {point['train']:.4f}, validation loss {point['val']:.4f}"
                         for point in points)
        else:
            lines.append('Waiting for the first saved epoch; the trainer writes history after each epoch.')
        return dict(total=self.total, points=points, lines=lines, watching=True,
                    status='Complete' if complete else 'Watching run',
                    stage='Training complete' if complete else 'Waiting for training results',
                    details=f'{self.run.name} · {self.total} training rounds',
                    elapsed=max(0, (self.preview.stat().st_mtime if complete else time.time()) - self.started),
                    preview=complete)


def main():
    # Launch the existing project manager so its checkpoint and logging behavior is preserved.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('quick', 'full'), default='quick')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/processed/yolo_real')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--continuous', action='store_true', help='Train indefinitely until Ctrl+C')
    parser.add_argument('--tile', action='store_true', help='Train and validate on image tiles')
    parser.add_argument('--patience', type=int, help='Stop after this many unimproved validation epochs')
    parser.add_argument('--verified-backgrounds', type=Path, help='Reviewed empty training images')
    parser.add_argument('--size', type=int, default=256, help='Input image size')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--human-weight', type=float, default=2.0)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--watch-run', type=Path, help='Watch an existing training run without launching training')
    args = parser.parse_args()
    if args.continuous and args.epochs is not None:
        parser.error('--continuous and --epochs cannot be combined')
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.patience is not None and args.patience < 1:
        parser.error('--patience must be positive')
    data = args.data.expanduser().resolve()
    if not (ROOT / 'project_manager.py').is_file():
        parser.error('Place live_dashboard.py beside project_manager.py')
    if args.watch_run:
        run = args.watch_run.resolve()
        if not run.is_relative_to((ROOT / 'training_runs').resolve()) or not (run / 'config.json').is_file():
            parser.error('--watch-run must point to an existing training run with config.json')
        monitor = SavedRunMonitor(run)
    else:
        if not all((data / kind / split).is_dir() for kind in ('images', 'labels') for split in ('train', 'val')):
            parser.error(f'Prepared images/labels not found under {data}; supply --data with the yolo_real path')
        monitor = Monitor('continuous' if args.continuous else (args.epochs or (2 if args.mode == 'quick' else 10)))

    # Bind only to this computer. The browser reads state; it cannot execute commands.
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path == '/':
                body, mime = PAGE.encode(), 'text/html; charset=utf-8'
            elif path == '/state':
                body, mime = json.dumps(monitor.snapshot(), allow_nan=False).encode(), 'application/json'
            elif path == '/preview' and monitor.preview and monitor.preview.is_file():
                body, mime = monitor.preview.read_bytes(), 'image/png'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{server.server_port}'
    if args.watch_run:
        print(f'Live dashboard: {url}\nWatching {run.name}. Ctrl+C closes this dashboard.', flush=True)
        try:
            if not args.no_browser:
                webbrowser.open(url)
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
            server.server_close()
        return
    print(f'Live dashboard: {url}\nThis starts a NEW training run. Ctrl+C stops it.', flush=True)
    cmd = [sys.executable, '-u', str(ROOT / 'project_manager.py'), 'train',
           '--mode', args.mode, '--data', str(data), '--device', args.device,
           '--size', str(args.size), '--batch', str(args.batch), '--human-weight', str(args.human_weight)]
    if args.continuous:
        cmd += ['--continuous']
    if args.tile:
        cmd += ['--tile']
    if args.patience:
        cmd += ['--patience', str(args.patience)]
    if args.verified_backgrounds:
        cmd += ['--verified-backgrounds', str(args.verified_backgrounds)]
    if args.epochs:
        cmd += ['--epochs', str(args.epochs)]
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    child = None
    try:
        # A Windows process group lets Ctrl+C stop both the manager and its training child.
        child = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding='utf-8', errors='replace', env=env,
                                 creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
        if not args.no_browser:
            webbrowser.open(url)
        for line in child.stdout:
            line = line.rstrip('\r\n')
            print(line, flush=True)
            monitor.record(line)
        code = child.wait()
        with monitor.lock:
            monitor.finished = time.monotonic()
            monitor.status = 'Complete' if code == 0 else 'Failed'
            monitor.stage = ('Training saved — review the preview and living report' if code == 0
                             else f'Training exited with code {code} — check live activity')
        print('Dashboard remains available. Press Ctrl+C to close it.', flush=True)
        threading.Event().wait()
    except KeyboardInterrupt:
        print('\nClosing dashboard.', flush=True)
    finally:
        if child is not None and child.poll() is None:
            if os.name == 'nt':
                child.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                child.terminate()
            child.wait()
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
