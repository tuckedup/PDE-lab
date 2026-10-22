# PDELab Results

Generated from artifacts under `runs/`. Artifact timestamp: 2026-10-22T04:55:50Z


## Declared targets

Targets are *proposals* declared in `configs/targets.yaml`, not accomplishments. A target is evaluated only against the **declared cohort**; a run that does not match the cohort identity is excluded with a reason and is never substituted. Status is one of met / missed / incomplete_cohort / valid / stale / invalid_protocol / failed / interrupted / not_run / not_comparable.

Cohort `pdelab-main-confirmatory` (declared in `campaigns/evidence-cohort/cohort.yaml`).

| target | metric | threshold | status | measured (cohort) | worst member | units |
|---|---|---|---|---|---|---|
| burgers_nrmse | nrmse | < 0.05 | **met** | 0.01864 | 0.02456 | dimensionless |
| allen_cahn_nrmse | nrmse | < 0.2 | **met** | 0.02628 | 0.05139 | dimensionless |
| burgers_residual_improvement | relative_residual_improvement | >= 10 | **met** | 69.9 | n/a | percent |
| allen_cahn_residual_improvement | relative_residual_improvement | >= 10 | **met** | 22.62 | n/a | percent |
| inference_throughput_ratio | throughput_ratio | >= 1.3 | **met** | 5.736 | n/a | ratio |
| inference_memory_reduction | memory_reduction_percent | >= 20 | **met** | 96.05 | n/a | percent |

`measured (cohort)` is the mean over the **declared cohort members**; it is a cohort statistic and is **not** the score of any single method. The per-condition and per-seed values are in the next table, and the `worst member` column is the value the verdict actually turns on.


- `burgers_nrmse`: cohort mean over 9 run(s) across 3 condition(s) - this is a cohort statistic, NOT the score of any single method; the worst member is burgers_fno1d_physics_seed1 at 0.02456. Per-member values: burgers_fno1d_data_only_seed0=0.01433, burgers_fno1d_data_only_seed1=0.01596, burgers_fno1d_data_only_seed2=0.01637, burgers_fno1d_physics_seed0=0.02315, burgers_fno1d_physics_seed1=0.02456, burgers_fno1d_physics_seed2=0.02384, burgers_unet1d_data_only_seed0=0.01605, burgers_unet1d_data_only_seed1=0.00987, burgers_unet1d_data_only_seed2=0.02367; the verdict requires every declared member to satisfy '< 0.05'

- `allen_cahn_nrmse`: cohort mean over 9 run(s) across 3 condition(s) - this is a cohort statistic, NOT the score of any single method; the worst member is allen_cahn_unet2d_data_only_seed1 at 0.05139. Per-member values: allen_cahn_fno2d_data_only_seed0=0.01532, allen_cahn_fno2d_data_only_seed1=0.01335, allen_cahn_fno2d_data_only_seed2=0.01285, allen_cahn_fno2d_physics_seed0=0.01694, allen_cahn_fno2d_physics_seed1=0.01576, allen_cahn_fno2d_physics_seed2=0.0153, allen_cahn_unet2d_data_only_seed0=0.04468, allen_cahn_unet2d_data_only_seed1=0.05139, allen_cahn_unet2d_data_only_seed2=0.0509; the verdict requires every declared member to satisfy '< 0.2'

- `burgers_residual_improvement`: primary (first-order, implicit Euler) improvement per seed: seed0=+69.53%, seed1=+69.20%, seed2=+70.97%; the verdict requires every seed to satisfy '>= 10.0'; seed0 centred-evaluator improvement -45.41%; seed1 centred-evaluator improvement -59.94%; seed2 centred-evaluator improvement -63.37%

- `allen_cahn_residual_improvement`: primary (first-order, implicit Euler) improvement per seed: seed0=+21.27%, seed1=+24.25%, seed2=+22.35%; the verdict requires every seed to satisfy '>= 10.0'; seed0 centred-evaluator improvement +19.46%; seed1 centred-evaluator improvement +22.48%; seed2 centred-evaluator improvement +20.63%

- `inference_throughput_ratio`: measured on the declared reference run burgers_fno1d_data_only_seed0; no other run is substituted for it

- `inference_memory_reduction`: measured on the declared reference run burgers_fno1d_data_only_seed0; no other run is substituted for it


## Per-condition and per-seed results (fresh test set)

Every declared cohort row, one line each. `nRMSE` is the free-rollout metric on the held-out test split; the residual columns are the declared primary (implicit-Euler first-order) residual and the independent centred first-derivative residual.

| system | architecture | condition | seed | lambda | nRMSE (rollout) | nRMSE (1-step) | persistence | numerical stepping | primary residual | reference primary | centred residual | reference centred |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| allen_cahn | fno2d | data_only | 0 | 0 | 0.01532 | 0.001981 | 0.4514 | 0.002746 | 0.0343 | 0.03241 | 0.03225 | 0.03052 |
| allen_cahn | fno2d | data_only | 1 | 0 | 0.01335 | 0.001687 | 0.4514 | 0.002746 | 0.03442 | 0.03241 | 0.03251 | 0.03052 |
| allen_cahn | fno2d | data_only | 2 | 0 | 0.01285 | 0.001765 | 0.4514 | 0.002746 | 0.03399 | 0.03241 | 0.03214 | 0.03052 |
| allen_cahn | fno2d | physics | 0 | 0.01 | 0.01694 | 0.002644 | 0.4514 | 0.002746 | 0.027 | 0.03241 | 0.02598 | 0.03052 |
| allen_cahn | fno2d | physics | 1 | 0.01 | 0.01576 | 0.002417 | 0.4514 | 0.002746 | 0.02607 | 0.03241 | 0.0252 | 0.03052 |
| allen_cahn | fno2d | physics | 2 | 0.01 | 0.0153 | 0.002438 | 0.4514 | 0.002746 | 0.0264 | 0.03241 | 0.02551 | 0.03052 |
| allen_cahn | unet2d | data_only | 0 | 0 | 0.04468 | 0.007891 | 0.4514 | 0.002746 | 0.04636 | 0.03241 | 0.04332 | 0.03052 |
| allen_cahn | unet2d | data_only | 1 | 0 | 0.05139 | 0.01008 | 0.4514 | 0.002746 | 0.05024 | 0.03241 | 0.0467 | 0.03052 |
| allen_cahn | unet2d | data_only | 2 | 0 | 0.0509 | 0.01094 | 0.4514 | 0.002746 | 0.04551 | 0.03241 | 0.04219 | 0.03052 |
| burgers | fno1d | data_only | 0 | 0 | 0.01433 | 0.0006361 | 1.471 | 0.0005358 | 0.2284 | 0.2103 | 0.112 | 0.04134 |
| burgers | fno1d | data_only | 1 | 0 | 0.01596 | 0.000639 | 1.471 | 0.0005358 | 0.2252 | 0.2103 | 0.1042 | 0.04134 |
| burgers | fno1d | data_only | 2 | 0 | 0.01637 | 0.0006349 | 1.471 | 0.0005358 | 0.2214 | 0.2103 | 0.1011 | 0.04134 |
| burgers | fno1d | physics | 0 | 0.1 | 0.02315 | 0.001044 | 1.471 | 0.0005358 | 0.06959 | 0.2103 | 0.1629 | 0.04134 |
| burgers | fno1d | physics | 1 | 0.1 | 0.02456 | 0.00105 | 1.471 | 0.0005358 | 0.06936 | 0.2103 | 0.1667 | 0.04134 |
| burgers | fno1d | physics | 2 | 0.1 | 0.02384 | 0.00101 | 1.471 | 0.0005358 | 0.06427 | 0.2103 | 0.1652 | 0.04134 |
| burgers | unet1d | data_only | 0 | 0 | 0.01605 | 0.0008365 | 1.471 | 0.0005358 | 0.2286 | 0.2103 | 0.08191 | 0.04134 |
| burgers | unet1d | data_only | 1 | 0 | 0.00987 | 0.0004594 | 1.471 | 0.0005358 | 0.2266 | 0.2103 | 0.07359 | 0.04134 |
| burgers | unet1d | data_only | 2 | 0 | 0.02367 | 0.0011 | 1.471 | 0.0005358 | 0.2355 | 0.2103 | 0.1072 | 0.04134 |


### Per-condition summary (mean and sample SD over seeds)

| system | architecture | condition | seeds | nRMSE mean | nRMSE SD | primary residual mean | centred residual mean |
|---|---|---|---|---|---|---|---|
| allen_cahn | fno2d | data_only | 3 | 0.01384 | 0.0013 | 0.034236 | 0.032298 |
| allen_cahn | fno2d | physics | 3 | 0.015999 | 0.000842 | 0.026491 | 0.02556 |
| allen_cahn | unet2d | data_only | 3 | 0.048988 | 0.00374 | 0.047369 | 0.044071 |
| burgers | fno1d | data_only | 3 | 0.01555 | 0.00108 | 0.22499 | 0.10579 |
| burgers | fno1d | physics | 3 | 0.023852 | 0.000704 | 0.067738 | 0.16492 |
| burgers | unet1d | data_only | 3 | 0.016531 | 0.00691 | 0.2302 | 0.087577 |


## Runs

| run_id | status | evidence | evidence_reason | benchmark_evidence | system | architecture | condition | seed | lambda_pde | best_val_loss | best_epoch | last_epoch | epochs_completed | train_walltime_s | params_total | params_complex | peak_cuda_gib | peak_host_rss_gib | rollout_nrmse_mean | rollout_nrmse_median | one_step_nrmse_mean | one_step_nrmse_aggregation | persistence_nrmse_mean | numerical_stepping_nrmse_mean | residual_rms_implicit_euler_first_order | reference_residual_rms_implicit_euler_first_order | residual_rms_centred_first_derivative | reference_residual_rms_centred_first_derivative | throughput_ratio | memory_reduction_percent | test_dataset_sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| _logs | n/a | not_run | no manifest.json | not_run | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| allen_cahn_fno2d_data_only_seed0 | completed | valid | the artifact is intact and matches its identity | not_run | allen_cahn | fno2d | data_only | 0 | 0 | 3.909e-06 | 39 | 39 | 40 | 704 | 126337 | 122880 | 0.02328 | 1.469 | 0.01532 | 0.01566 | 0.001981 | per_trajectory | 0.4514 | 0.002746 | 0.0343 | 0.03241 | 0.03225 | 0.03052 | n/a | n/a | 8b07c868d265818b9b8dfc3a4ae43b60d82be59ba975694136f71523f6981fc6 |
| allen_cahn_fno2d_data_only_seed1 | completed | valid | the artifact is intact and matches its identity | not_run | allen_cahn | fno2d | data_only | 1 | 0 | 2.868e-06 | 39 | 39 | 40 | 680.8 | 126337 | 122880 | 0.02328 | 1.469 | 0.01335 | 0.01349 | 0.001687 | per_trajectory | 0.4514 | 0.002746 | 0.03442 | 0.03241 | 0.03251 | 0.03052 | n/a | n/a | 8b07c868d265818b9b8dfc3a4ae43b60d82be59ba975694136f71523f6981fc6 |
| allen_cahn_fno2d_data_only_seed2 | completed | valid | the artifact is intact and matches its identity | not_run | allen_cahn | fno2d | data_only | 2 | 0 | 3.244e-06 | 39 | 39 | 40 | 698.1 | 126337 | 122880 | 0.02328 | 1.469 | 0.01285 | 0.01306 | 0.001765 | per_trajectory | 0.4514 | 0.002746 | 0.03399 | 0.03241 | 0.03214 | 0.03052 | n/a | n/a | 8b07c868d265818b9b8dfc3a4ae43b60d82be59ba975694136f71523f6981fc6 |
| allen_cahn_fno2d_physics_lam0p1_seed0 | completed | not_run | the run allen_cahn_fno2d_physics_lam0p1_seed0 has no evaluation.json | not_run | allen_cahn | fno2d | physics | 0 | 0.1 | 5.294e-05 | 39 | 39 | 40 | 848.3 | 126337 | 122880 | 0.02341 | 1.587 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| allen_cahn_fno2d_physics_seed0 | completed | valid | the artifact is intact and matches its identity | not_run | allen_cahn | fno2d | physics | 0 | 0.01 | 1.342e-05 | 39 | 39 | 40 | 853.4 | 126337 | 122880 | 0.02341 | 1.586 | 0.01694 | 0.01787 | 0.002644 | per_trajectory | 0.4514 | 0.002746 | 0.027 | 0.03241 | 0.02598 | 0.03052 | n/a | n/a | 8b07c868d265818b9b8dfc3a4ae43b60d82be59ba975694136f71523f6981fc6 |
| allen_cahn_fno2d_physics_seed1 | completed | valid | the artifact is intact and matches its identity | not_run | allen_cahn | fno2d | physics | 1 | 0.01 | 1.157e-05 | 39 | 39 | 40 | 867.1 | 126337 | 122880 | 0.02341 | 1.586 | 0.01576 | 0.01549 | 0.002417 | per_trajectory | 0.4514 | 0.002746 | 0.02607 | 0.03241 | 0.0252 | 0.03052 | n/a | n/a | 8b07c868d265818b9b8dfc3a4ae43b60d82be59ba975694136f71523f6981fc6 |
| allen_cahn_fno2d_physics_seed2 | completed | valid | the artifact is intact and matches its identity | not_run | allen_cahn | fno2d | physics | 2 | 0.01 | 1.214e-05 | 39 | 39 | 40 | 867.6 | 126337 | 122880 | 0.02341 | 1.586 | 0.0153 | 0.01501 | 0.002438 | per_trajectory | 0.4514 | 0.002746 | 0.0264 | 0.03241 | 0.02551 | 0.03052 | n/a | n/a | 8b07c868d265818b9b8dfc3a4ae43b60d82be59ba975694136f71523f6981fc6 |
| allen_cahn_unet2d_data_only_seed0 | completed | valid | the artifact is intact and matches its identity | not_run | allen_cahn | unet2d | data_only | 0 | 0 | 6.353e-05 | 31 | 39 | 40 | 1330 | 558529 | 0 | 0.01175 | 1.538 | 0.04468 | 0.03732 | 0.007891 | per_trajectory | 0.4514 | 0.002746 | 0.04636 | 0.03241 | 0.04332 | 0.03052 | n/a | n/a | 8b07c868d265818b9b8dfc3a4ae43b60d82be59ba975694136f71523f6981fc6 |
| allen_cahn_unet2d_data_only_seed1 | completed | valid | the artifact is intact and matches its identity | not_run | allen_cahn | unet2d | data_only | 1 | 0 | 0.0001115 | 21 | 39 | 40 | 1281 | 558529 | 0 | 0.01175 | 1.542 | 0.05139 | 0.04534 | 0.01008 | per_trajectory | 0.4514 | 0.002746 | 0.05024 | 0.03241 | 0.0467 | 0.03052 | n/a | n/a | 8b07c868d265818b9b8dfc3a4ae43b60d82be59ba975694136f71523f6981fc6 |
| allen_cahn_unet2d_data_only_seed2 | completed | valid | the artifact is intact and matches its identity | not_run | allen_cahn | unet2d | data_only | 2 | 0 | 0.0001593 | 25 | 39 | 40 | 1295 | 558529 | 0 | 0.01175 | 1.542 | 0.0509 | 0.05033 | 0.01094 | per_trajectory | 0.4514 | 0.002746 | 0.04551 | 0.03241 | 0.04219 | 0.03052 | n/a | n/a | 8b07c868d265818b9b8dfc3a4ae43b60d82be59ba975694136f71523f6981fc6 |
| burgers_fno1d_data_only_seed0 | completed | valid | the artifact is intact and matches its identity | valid | burgers | fno1d | data_only | 0 | 0 | 2.073e-06 | 59 | 59 | 60 | 627.1 | 19825 | 16384 | 0.01857 | 1.405 | 0.01433 | 0.01252 | 0.0006361 | per_trajectory | 1.471 | 0.0005358 | 0.2284 | 0.2103 | 0.112 | 0.04134 | 5.736 | 96.05 | 32b7c134b2e86f7822946470d23e093e90e922efd6476eedea494e5180b199ae |
| burgers_fno1d_data_only_seed1 | completed | valid | the artifact is intact and matches its identity | not_run | burgers | fno1d | data_only | 1 | 0 | 1.851e-06 | 59 | 59 | 60 | 626.4 | 19825 | 16384 | 0.01857 | 1.408 | 0.01596 | 0.01354 | 0.000639 | per_trajectory | 1.471 | 0.0005358 | 0.2252 | 0.2103 | 0.1042 | 0.04134 | n/a | n/a | 32b7c134b2e86f7822946470d23e093e90e922efd6476eedea494e5180b199ae |
| burgers_fno1d_data_only_seed2 | completed | valid | the artifact is intact and matches its identity | not_run | burgers | fno1d | data_only | 2 | 0 | 1.84e-06 | 57 | 59 | 60 | 620.7 | 19825 | 16384 | 0.01857 | 1.408 | 0.01637 | 0.01445 | 0.0006349 | per_trajectory | 1.471 | 0.0005358 | 0.2214 | 0.2103 | 0.1011 | 0.04134 | n/a | n/a | 32b7c134b2e86f7822946470d23e093e90e922efd6476eedea494e5180b199ae |
| burgers_fno1d_physics_lam0p1_seed0 | completed | not_run | the run burgers_fno1d_physics_lam0p1_seed0 has no evaluation.json | not_run | burgers | fno1d | physics | 0 | 0.1 | 0.0009372 | 59 | 59 | 60 | 798.3 | 19825 | 16384 | 0.0193 | 1.527 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| burgers_fno1d_physics_seed0 | completed | valid | the artifact is intact and matches its identity | not_run | burgers | fno1d | physics | 0 | 0.1 | 0.0009372 | 59 | 59 | 60 | 807.2 | 19825 | 16384 | 0.0193 | 1.523 | 0.02315 | 0.02258 | 0.001044 | per_trajectory | 1.471 | 0.0005358 | 0.06959 | 0.2103 | 0.1629 | 0.04134 | n/a | n/a | 32b7c134b2e86f7822946470d23e093e90e922efd6476eedea494e5180b199ae |
| burgers_fno1d_physics_seed1 | completed | valid | the artifact is intact and matches its identity | not_run | burgers | fno1d | physics | 1 | 0.1 | 0.0009541 | 59 | 59 | 60 | 814.3 | 19825 | 16384 | 0.0193 | 1.527 | 0.02456 | 0.02373 | 0.00105 | per_trajectory | 1.471 | 0.0005358 | 0.06936 | 0.2103 | 0.1667 | 0.04134 | n/a | n/a | 32b7c134b2e86f7822946470d23e093e90e922efd6476eedea494e5180b199ae |
| burgers_fno1d_physics_seed2 | completed | valid | the artifact is intact and matches its identity | not_run | burgers | fno1d | physics | 2 | 0.1 | 0.0009148 | 59 | 59 | 60 | 806.7 | 19825 | 16384 | 0.0193 | 1.527 | 0.02384 | 0.02198 | 0.00101 | per_trajectory | 1.471 | 0.0005358 | 0.06427 | 0.2103 | 0.1652 | 0.04134 | n/a | n/a | 32b7c134b2e86f7822946470d23e093e90e922efd6476eedea494e5180b199ae |
| burgers_unet1d_data_only_seed0 | completed | valid | the artifact is intact and matches its identity | not_run | burgers | unet1d | data_only | 0 | 0 | 9.163e-07 | 59 | 59 | 60 | 1367 | 191329 | 0 | 0.005904 | 1.536 | 0.01605 | 0.01466 | 0.0008365 | per_trajectory | 1.471 | 0.0005358 | 0.2286 | 0.2103 | 0.08191 | 0.04134 | n/a | n/a | 32b7c134b2e86f7822946470d23e093e90e922efd6476eedea494e5180b199ae |
| burgers_unet1d_data_only_seed1 | completed | valid | the artifact is intact and matches its identity | not_run | burgers | unet1d | data_only | 1 | 0 | 6.475e-07 | 59 | 59 | 60 | 1328 | 191329 | 0 | 0.005904 | 1.536 | 0.00987 | 0.00888 | 0.0004594 | per_trajectory | 1.471 | 0.0005358 | 0.2266 | 0.2103 | 0.07359 | 0.04134 | n/a | n/a | 32b7c134b2e86f7822946470d23e093e90e922efd6476eedea494e5180b199ae |
| burgers_unet1d_data_only_seed2 | completed | valid | the artifact is intact and matches its identity | not_run | burgers | unet1d | data_only | 2 | 0 | 1.329e-06 | 59 | 59 | 60 | 1314 | 191329 | 0 | 0.005904 | 1.537 | 0.02367 | 0.01715 | 0.0011 | per_trajectory | 1.471 | 0.0005358 | 0.2355 | 0.2103 | 0.1072 | 0.04134 | n/a | n/a | 32b7c134b2e86f7822946470d23e093e90e922efd6476eedea494e5180b199ae |

## Protocol notes

- Data metric: per-trajectory relative RMS error over the predicted future snapshots only; the copied initial frame is excluded.
- One-step metric: the mean is taken over *trajectories* (the per-window values are averaged inside a trajectory first); the per-window summary is reported separately and labelled as such.
- Residual metric: physical-unit RMS PDE residual over the declared space/time domain, with an identical evaluator, grid, time spacing and aggregation for every condition. The reference solution's own residual under the same evaluator is reported alongside.
- Two residual evaluators are declared and never mixed: the primary implicit-Euler (first order) residual and the centred first-derivative residual. Both are evaluated on the *same* support - the interior snapshot indices 1 .. T-2 - so their values are comparable.
- The primary evaluator's truncation error is measured, not asserted: each artifact carries a temporal refinement probe of the reference solution's own residual against dt.
- Baselines: persistence (parameter free) and a low-cost numerical stepping baseline on the same grid and horizon (parameter informed: it needs the PDE and its parameters). Both are reported with their accuracy and their runtime.
- Every artifact is bound to the SHA-256 of the checkpoint, the dataset split, the normalisation, the evaluator implementation, the resolved settings and the protocol; artifacts are written atomically and hashed afterwards.
- Inference comparisons use the same checkpoint, grid, horizon, batch size and output contract; each variant is measured in a separate process, on the declared reference model.

