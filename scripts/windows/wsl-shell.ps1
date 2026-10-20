<#
.SYNOPSIS
    Open the PDELab project directory in a WSL shell.

.DESCRIPTION
    Starts an interactive Ubuntu WSL2 shell in the project directory
    with the project's virtual environment already activated. The
    environment lives on the Linux filesystem; nothing is installed on
    Windows.

.EXAMPLE
    .\wsl-shell.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$ProjectDir = '/home/atulp/projects/pdelab'
$Distro     = 'Ubuntu'

wsl.exe -d $Distro --cd $ProjectDir -- bash -lc 'source .venv/bin/activate 2>/dev/null; export PATH="$HOME/.local/bin:$PATH"; echo "PDELab environment active in $(pwd)"; python --version; exec bash'
