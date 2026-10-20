# PDELab Implementation Plan

## Phase 1: Foundation (Current)
- [ ] Environment lock (`uv`, PyTorch, CUDA).
- [ ] Hardware limits configuration.
- [ ] Base CLI and logging.

## Phase 2: Numerical Solvers
- [ ] 1D Burgers (Periodic, Finite Volume/Spectral).
- [ ] 2D Allen-Cahn (Periodic, Finite Difference).
- [ ] Validation against analytic/mode-based solutions.

## Phase 3: Models & Physics
- [ ] FNO 1D/2D with complex weights.
- [ ] Periodic U-Net 1D/2D.
- [ ] Differentiable PDE residuals (Autograd/Functional).

## Phase 4: Evaluation & Benchmarking
- [ ] Multi-step rollout metrics (nRMSE).
- [ ] Throughput/Memory benchmarks.
- [ ] Automated reporting.
