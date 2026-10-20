# PDELab Results

Generated from artifacts under `runs/`. Artifact timestamp: 2026-10-20T04:55:19Z


## Declared targets

Targets are *proposals* declared in `configs/targets.yaml`, not accomplishments. Status is one of met / missed / not_run / not_comparable.

| target | metric | threshold | status | measured | units | run_id |
|---|---|---|---|---|---|---|
| burgers_nrmse | nrmse | < 0.05 | **met** | 0.03723 | dimensionless | burgers_fno1d_physics_seed0 |
| allen_cahn_nrmse | nrmse | < 0.2 | **met** | 0.04355 | dimensionless | allen_cahn_fno2d_physics_seed0 |
| burgers_residual_improvement | relative_residual_improvement | >= 10 | **met** | 88.08 | percent | burgers_fno1d_physics_seed0 |
| allen_cahn_residual_improvement | relative_residual_improvement | >= 10 | **met** | 20.64 | percent | allen_cahn_fno2d_physics_seed0 |
| inference_throughput_ratio | throughput_ratio | >= 1.3 | **met** | 6.108 | ratio | burgers_fno1d_data_only_seed0 |
| inference_memory_reduction | memory_reduction_percent | >= 20 | **missed** | 1.718 | percent | burgers_fno1d_data_only_seed0 |

- `burgers_nrmse`: per-condition rollout nRMSE: data_only=0.08132, physics=0.03723

- `allen_cahn_nrmse`: per-condition rollout nRMSE: data_only=0.04425, physics=0.04355

- `burgers_residual_improvement`: second-order (diagnostic) evaluator gives +23.82%; baseline residual 1.006, proposed 0.1199

- `allen_cahn_residual_improvement`: second-order (diagnostic) evaluator gives -2.60%; baseline residual 0.08147, proposed 0.06465

- `inference_throughput_ratio`: per-run values: allen_cahn_fno2d_data_only_seed0=6.78, allen_cahn_fno2d_physics_seed0=6.54, allen_cahn_unet2d_data_only_seed0=7.65, allen_cahn_unet2dw4_data_only_seed0=10.6, burgers_fno1d_data_only_seed0=6.11, burgers_fno1d_physics_seed0=5.84, burgers_unet1d_data_only_seed0=11.2, burgers_unet1dw4_data_only_seed0=11.6

- `inference_memory_reduction`: per-run values: allen_cahn_fno2d_data_only_seed0=7.36, allen_cahn_fno2d_physics_seed0=7.36, allen_cahn_unet2d_data_only_seed0=25.6, allen_cahn_unet2dw4_data_only_seed0=10.5, burgers_fno1d_data_only_seed0=1.72, burgers_fno1d_physics_seed0=1.72, burgers_unet1d_data_only_seed0=5.28, burgers_unet1dw4_data_only_seed0=5.35; the reduction depends strongly on the model size, because fixed costs (cuFFT workspace, allocator granularity) dominate the peak for the small models used here


## Runs

| run_id | status | system | architecture | condition | seed | lambda_pde | best_val_loss | best_epoch | train_walltime_s | params_total | params_complex | peak_cuda_gib | peak_host_rss_gib | rollout_nrmse_mean | rollout_nrmse_median | one_step_nrmse_mean | persistence_nrmse_mean | residual_rms | reference_residual_rms | residual_rms_order2 | throughput_ratio | memory_reduction_percent | test_dataset_sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| allen_cahn_fno2d_data_only_seed0 | completed | allen_cahn | fno2d | data_only | 0 | 0 | 4.919e-05 | 39 | 515.9 | 68993 | 65536 | 0.02152 | 1.446 | 0.04425 | 0.04317 | 0.005301 | 0.4146 | 0.08147 | 0.0607 | 0.2313 | 6.781 | 7.356 | 6ec8aca7f73021c9f0e17ccd9ba3a53799262c27a17cf636485f9a7abad5b3ae |
| allen_cahn_fno2d_physics_seed0 | completed | allen_cahn | fno2d | physics | 0 | 0.01 | 6.189e-05 | 39 | 641.5 | 68993 | 65536 | 0.0225 | 1.564 | 0.04355 | 0.04154 | 0.005086 | 0.4146 | 0.06465 | 0.0607 | 0.2373 | 6.541 | 7.356 | 6ec8aca7f73021c9f0e17ccd9ba3a53799262c27a17cf636485f9a7abad5b3ae |
| allen_cahn_unet2d_data_only_seed0 | completed | allen_cahn | unet2d | data_only | 0 | 0 | 7.501e-05 | 33 | 1333 | 558529 | 0 | 0.01175 | 1.524 | 0.04567 | 0.04572 | 0.007881 | 0.4146 | 0.04244 | 0.0607 | 0.2321 | 7.654 | 25.58 | 6ec8aca7f73021c9f0e17ccd9ba3a53799262c27a17cf636485f9a7abad5b3ae |
| allen_cahn_unet2dw4_data_only_seed0 | completed | allen_cahn | unet2d | data_only | 0 | 0 | 0.004556 | 20 | 1296 | 35377 | 0 | 0.001649 | 1.481 | 0.1325 | 0.1182 | 0.07313 | 0.4146 | 0.1603 | 0.0607 | 0.2794 | 10.59 | 10.51 | 6ec8aca7f73021c9f0e17ccd9ba3a53799262c27a17cf636485f9a7abad5b3ae |
| burgers_fno1d_data_only_seed0 | completed | burgers | fno1d | data_only | 0 | 0 | 8.036e-05 | 56 | 634.6 | 19825 | 16384 | 0.01857 | 1.407 | 0.08132 | 0.07857 | 0.00352 | 1.478 | 1.006 | 1.084 | 106.3 | 6.108 | 1.718 | b1ee09a87c905df335fa755dfd93aadb36a3c59c103c3cd3b943e290eb446e25 |
| burgers_fno1d_physics_seed0 | completed | burgers | fno1d | physics | 0 | 0.01 | 0.000234 | 56 | 799.6 | 19825 | 16384 | 0.0193 | 1.526 | 0.03723 | 0.03574 | 0.002745 | 1.478 | 0.1199 | 1.084 | 80.98 | 5.835 | 1.718 | b1ee09a87c905df335fa755dfd93aadb36a3c59c103c3cd3b943e290eb446e25 |
| burgers_unet1d_data_only_seed0 | completed | burgers | unet1d | data_only | 0 | 0 | 7.076e-05 | 59 | 1291 | 191329 | 0 | 0.005904 | 1.518 | 0.09511 | 0.06857 | 0.003981 | 1.478 | 2.246 | 1.084 | 225.7 | 11.18 | 5.285 | b1ee09a87c905df335fa755dfd93aadb36a3c59c103c3cd3b943e290eb446e25 |
| burgers_unet1dw4_data_only_seed0 | completed | burgers | unet1d | data_only | 0 | 0 | 0.05728 | 39 | 1292 | 12409 | 0 | 0.001166 | 1.498 | 0.6831 | 0.5954 | 0.2868 | 1.478 | 6.007 | 1.084 | 555.4 | 11.55 | 5.345 | b1ee09a87c905df335fa755dfd93aadb36a3c59c103c3cd3b943e290eb446e25 |

## Protocol notes

- Data metric: per-trajectory relative RMS error over the predicted future snapshots only; the copied initial frame is excluded.
- Residual metric: physical-unit RMS PDE residual over the declared space/time domain, with an identical evaluator, grid, time spacing and aggregation for every condition. The reference solution's own residual under the same evaluator is reported alongside.
- The primary residual evaluator uses a forward-difference time derivative; a separately declared second-order (centred) evaluator is reported as a diagnostic and is never mixed with the primary one.
- Inference comparisons use the same checkpoint, grid, horizon, batch size and output contract; each variant is measured in a separate process.

