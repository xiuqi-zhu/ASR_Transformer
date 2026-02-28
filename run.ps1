# Helper script for Windows: sets PYTHONPATH to transformer-from-scratch and runs a command.
# Usage: .\run.ps1 python main.py --mode train ...
# Edit $TransformerScratchDir to your transformer-from-scratch clone path.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TransformerScratchDir = if ($env:TRANSFORMER_SCRATCH_DIR) { $env:TRANSFORMER_SCRATCH_DIR } else { Join-Path (Split-Path -Parent $ScriptDir) "transformer-from-scratch" }

if (-not (Test-Path $TransformerScratchDir)) {
    Write-Host "Error: transformer-from-scratch not found at: $TransformerScratchDir" -ForegroundColor Red
    Write-Host "Clone it: git clone https://github.com/xiuqi-zhu/transformer-from-scratch.git"
    Write-Host "Or set: `$env:TRANSFORMER_SCRATCH_DIR = 'C:\path\to\transformer-from-scratch'"
    exit 1
}

$env:PYTHONPATH = "$TransformerScratchDir" + $(if ($env:PYTHONPATH) { ";" + $env:PYTHONPATH } else { "" })
Set-Location $ScriptDir
if ($args.Count -gt 0) {
    & @args
} else {
    Write-Host "Usage: .\run.ps1 python main.py [options]"
}
