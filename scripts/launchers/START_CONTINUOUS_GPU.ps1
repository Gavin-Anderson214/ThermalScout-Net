$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $project

$activeRun = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'live_dashboard\.py.*--continuous|thermalscout\.train_detector.*--continuous' }
if ($activeRun) {
    Write-Host 'Continuous training is already running. Open its existing PowerShell window or dashboard.'
    return
}

& py -3.12 -c "import torch,sys; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'); sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) {
    throw 'CUDA is unavailable. Training was not started.'
}

Write-Host 'Starting continuous ThermalScout-Net training on the GPU.'
Write-Host 'A browser dashboard will open. This PowerShell window lists completed epochs, training loss, and validation loss.'
Write-Host 'The best weights and loss history are saved under training_runs after every completed epoch.'
& py -3.12 -u live_dashboard.py --mode full --continuous --size 512 --batch 4 --human-weight 4 --device cuda
