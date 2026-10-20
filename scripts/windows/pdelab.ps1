<#
.SYNOPSIS
    Run PDELab inside Ubuntu WSL2 from Windows PowerShell.

.DESCRIPTION
    OpenCode runs on the Windows host; all computation runs inside WSL2.
    This wrapper forwards a command to the project directory in the
    Linux environment, where the isolated .venv lives. Nothing is
    installed on Windows and no project file is duplicated there.

.PARAMETER Command
    The pdelab CLI arguments, e.g. 'doctor' or 'train --config X --profile prototype'.

.EXAMPLE
    .\pdelab.ps1 doctor
    .\pdelab.ps1 data generate --config configs/pdes/burgers.yaml --profile prototype
    .\pdelab.ps1 report
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

$ErrorActionPreference = 'Stop'

$ProjectDir = '/home/atulp/projects/pdelab'
$Distro     = 'Ubuntu'
$Python     = '.venv/bin/python'

if ($Command.Count -eq 0) {
    Write-Host "Usage: .\pdelab.ps1 <pdelab command> [args...]" -ForegroundColor Yellow
    Write-Host "  e.g. .\pdelab.ps1 doctor"
    Write-Host "  e.g. .\pdelab.ps1 train --config configs/experiments/burgers_fno1d_data_only_seed0.yaml --profile prototype"
    exit 2
}

# Check the distribution exists before trying to use it.
$distros = (wsl.exe --list --quiet 2>$null) | ForEach-Object { $_.Trim() } | Where-Object { $_ }
if ($distros -notcontains $Distro) {
    Write-Error "WSL distribution '$Distro' not found. Available: $($distros -join ', ')"
    exit 1
}

# Verify the project directory and the environment exist.
wsl.exe -d $Distro --cd $ProjectDir -- test -x "$Python" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "The Linux environment is not ready in $ProjectDir." -ForegroundColor Red
    Write-Host "Create it once with:" -ForegroundColor Yellow
    Write-Host "  wsl -d $Distro --cd $ProjectDir -- bash -lc 'curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=\`$HOME/.local/bin:\`$PATH && uv venv --python 3.11 && uv pip install torch --index-url https://download.pytorch.org/whl/cu121 && uv pip install -e .'"
    exit 1
}

Write-Host "=== wsl -d $Distro --cd $ProjectDir :: pdelab $($Command -join ' ') ===" -ForegroundColor DarkGray
wsl.exe -d $Distro --cd $ProjectDir -- $Python -m pdelab.cli @Command
exit $LASTEXITCODE
