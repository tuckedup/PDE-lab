# PDELab Agent Instructions

This project implements a reproducible local PyTorch research prototype comparing Fourier Neural Operators (FNO) and U-Nets with physics-regularized learning.

## Execution Boundary
- **Host:** Windows 11 (atulp)
- **Runtime:** Ubuntu WSL2 (`/home/atulp/projects/pdelab`)
- **Tools:** `uv` for Python management, `wsl` for execution.
- **Project Files:** Edited via Windows UNC path `\\wsl.localhost\Ubuntu\home\atulp\projects\pdelab`.

## Resource Constraints (Atul's Laptop)
- **GPU:** NVIDIA RTX 4060 (8GB VRAM). Max allocation: 4.5GB.
- **RAM:** ~15GB Host, ~7GB WSL. Max RSS: 3.5GB.
- **Concurrency:** 1 (Sequential experiments only).

## Verification Workflow
1. `pdelab doctor`: Check environment and resources.
2. `pdelab data generate --profile smoke`: Small verification dataset.
3. `pdelab train --profile smoke`: Verify gradients and code integrity.
4. `pdelab benchmark`: Matched inference comparison.

## Result Honesty
- Record all seeds.
- Report failures and OOMs.
- Do not backdate results.
