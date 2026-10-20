# PDELab Implementation Plan

Status markers: `[x]` done, `[~]` partially done, `[ ]` not started.
Current state: see `docs/STATUS.md` for what is measured and outstanding.

## Phase 1 - Foundation `[x]`

- [x] Windows/WSL2 boundary established; OpenCode on Windows, all
      computation in Ubuntu WSL2.
- [x] Isolated Linux `.venv` via `uv`; PyTorch 2.14.1+cu130, CUDA
      forward/backward verified on the RTX 4060.
- [x] `AGENTS.md` written with the boundary, the resource limits and
      the honesty rules.
- [x] `configs/hardware/atul_laptop.yaml` with conservative limits.
- [x] Resource preflight and the project job lock.
- [x] Base CLI (`doctor`, `resources check`, `data generate`,
      `solvers validate`, `train`, `evaluate`, `sweep`, `benchmark`,
      `report`) and the three execution profiles.

## Phase 2 - Numerical solvers `[x]`

- [x] 1D Burgers: WENO5-JS finite volume + SSPRK(3,3), plus a
      MUSCL-Rusanov and a dealiased spectral cross-check.
- [x] 2D Allen-Cahn: pseudo-spectral with exact Fourier diffusion and
      Strang splitting.
- [x] Validation suite: 12 quantitative checks per system, all
      passing, including two manufactured-solution tests with recorded
      forcing terms and a measured reference-solution own error.
- [x] Deterministic generation from seeds, verified to reproduce a
      stored trajectory to float32 precision.

## Phase 3 - Models, physics and evaluation `[x]`

- [x] FNO 1D/2D with learned complex spectral mixing; modes, shapes,
      Nyquist limits and gradients tested.
- [x] Periodic U-Net 1D/2D (circular padding, nearest-neighbour
      upsampling); periodicity and shape alignment tested.
- [x] One-step and free-rollout evaluation; nRMSE, persistence
      baseline, physical residual with a measured floor, and a
      separately declared second-order diagnostic evaluator.
- [x] Physics residual operators and the composite objective; the
      physics-gradient check is recorded in every physics run.

## Phase 4 - Prototype study `[~]`

- [x] Prototype datasets for both systems (64/16/16 per system).
- [x] Eight training runs at seed 0: FNO data-only, FNO physics and
      U-Net data-only for each system, plus two near
      parameter-matched U-Nets.
- [x] Evaluation and matched inference benchmark for every run.
- [x] Generated result tables, plots and the full documentation set.
- [ ] **Three independent training seeds** (only seed 0 so far).
- [ ] Physics-regularized U-Net (configured, not run).
- [ ] The λ grid {0, 0.01, 0.1} as a validation-only search.
- [ ] The remaining ablations (spectral modes / width).

## Phase 5 - Confirmatory study `[ ]`

- [ ] Write the seed-1 and seed-2 experiment configs.
- [ ] Freeze the confirmatory dataset size *before* looking at its test
      results.
- [ ] Estimate the matrix cost and split it across sessions, or raise
      the sweep budget explicitly (the 18-run matrix is ~3.6 h at the
      measured cost, against a 120-minute declared sweep budget).
- [ ] Run the 2 PDEs x 3 conditions x 3 seeds matrix sequentially.
- [ ] Report every seed individually plus the mean and the spread, and
      state explicitly that three seeds give a limited uncertainty
      estimate.

## Phase 6 - Extensions `[ ]`

- [ ] A parameter-OOD or resolution-transfer benchmark with its own
      labels, not replacing the main held-out test.
- [ ] The maximum-throughput-at-best-batch profile, clearly labelled as
      a different workload.
- [ ] A `LICENSE` file before any redistribution.
