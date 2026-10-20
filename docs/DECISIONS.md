# Decisions and their reasons

Every entry records a choice that was *not* forced by the code, the
evidence behind it, and what was deliberately not done.

---

## D1 - The 2D horizon was extended after the first pilot

**When:** after the first 2D pilot, before the confirmatory matrix.

**Evidence:** with `T = 1.0` and `dt_obs = 0.05`, the persistence
baseline (copying the initial frame over the whole horizon) scored
nRMSE **0.197**, while the declared accuracy target was 0.20. The
brief requires a meaningful prediction horizon and explicitly asks to
avoid a configuration where simply copying the input nearly meets the
accuracy goal. The trivial baseline was within 2% of the target.

**Decision:** `T = 3.0`, `dt_obs = 0.1`. The measured persistence
baseline becomes **0.415**, more than twice the target.

**Why this direction:** the revision makes the benchmark **harder**, not
easier. It was not made to move a result in either direction, and the
revision is recorded in the config file, the pre-registration and the
PDE specification.

**Not done:** the *declared target* was not changed after seeing the
results. The 0.20 target still stands, and the 1D horizon was left
alone (its persistence baseline of 1.478 is 30x the target, so it was
already meaningful).

---

## D2 - WENO5 is the primary Burgers generator

**Evidence:** the minmod-limited MUSCL-Rusanov scheme showed a global
L2 convergence of order **1.00** and a 2.5% error against a converged
spectral solution at n = 1024 - far too large for a reference.

**Decision:** implement WENO5-JS as the primary generator and keep the
MUSCL scheme as an independent cross-check. WENO5's error at n = 256 in
the shock-forming regime is 3.9e-5, about 12x smaller.

**Not done:** the 2/3 de-aliasing rule was *not* extended to 1/6 for the
cubic nonlinearity in the Allen-Cahn solver, because a direct experiment
(identical results at `N/3` and `N/6`) showed that aliasing was not the
cause of the observed behaviour there.

---

## D3 - The U-Net decoder uses nearest-neighbour upsampling, not a transposed convolution

**Evidence:** a unit test asserting that a spatially constant field maps
to a spatially constant field failed. The cause is the well-known
checkerboard artefact of `ConvTranspose1d(kernel=2, stride=2)`, which
produces a period-2 alternation even for a constant input.

**Decision:** nearest-neighbour interpolation followed by a convolution.

**Why:** a checkerboard artefact is an implementation wart of the
baseline, not a property of the U-Net family, and it would unfairly
handicap the spatial baseline. The brief explicitly warns against
injecting a defect into the U-Net and then presenting the FNO's
advantage as a general result.

**Related, documented property:** a stride-2 U-Net is translation
covariant only for shifts that are multiples of `2 ** n_levels`; an odd
shift offsets the coarse features by half a cell. This is inherent to
strided U-Nets, is *not* a boundary-handling defect (the circular
padding is correct), and is asserted by the test suite.

---

## D4 - The inference targets are evaluated on a declared reference model

**Evidence:** the measured peak-memory reduction depends strongly on the
model size (1.7% for the 1D FNO, 7.4% for the 2D FNO, 25.6% for the
2D U-Net), because fixed costs - the cuFFT workspace and allocator
granularity - dominate the peak for the small models used here.

**Decision:** `configs/targets.yaml` declares
`inference_reference_model: burgers_fno1d_data_only_seed0`, and the
target table reports that model's value, with the per-run values for
every other model shown alongside.

**Why:** reporting the maximum over models would let the result be
inflated by selecting a favourable checkpoint.

---

## D5 - The physics regulariser is applied to the predicted state only

**Decision:** the residual is formed between the **predicted** state at
`t + dt` and the given state at `t`:

```
R = (u_pred - u_prev)/dt + flux_x(u_pred) - nu * uxx(u_pred)
```

Gradients flow through `u_pred` only. The ground-truth trajectory is
never used as the residual carrier, and the residual is never computed
on a detached prediction.

**Verification:** every physics run records a `physics_grad_check` in
its manifest: the gradient of the physics loss alone is computed and
inspected for finiteness and non-zero-ness. All physics runs pass
(`grads_finite: true`, `grads_nonzero: true`, L1 sum 7149-10080 across
18 parameter tensors).

---

## D6 - No trainable IC or boundary penalty is used

Both benchmarks are periodic and the initial condition is supplied
exactly as an input channel.

* A trainable IC constraint would be redundant, so `lambda_ic` is 0 and
  `use_ic` is false.
* The periodic boundary is enforced **structurally** - by the spectral
  operators and by circular padding in the U-Net - not by an artificial
  boundary penalty. This is documented as a structural enforcement
  rather than reported as the outcome of a penalty term.

`lambda_bc` is 0 and `use_bc` is false for the same reason.

---

## D7 - The residual metric is reported with a measured floor and a second evaluator

**Decision:** two residual evaluators are reported, under distinct
names, and are never mixed:

* **primary** - forward-difference time derivative. Its O(dt * u_tt)
  truncation error means an exact PDE solution sampled at `dt` still
  shows a non-zero residual. That floor is measured and reported for
  every run (1.08 for Burgers, 0.061 for Allen-Cahn).
* **diagnostic (second order)** - centred second difference, which
  cancels the O(dt) term. It is reported so that a residual improvement
  that is specific to one operator becomes visible.

**Why it matters here:** on Burgers the physics model improves the
primary residual by 88% and the diagnostic residual by 24%; on
Allen-Cahn it improves the primary residual by 21% but the diagnostic
residual by **-2.6%**. Reporting only the primary number would
overstate the Allen-Cahn result.

**Guard against degenerate solutions:** a constant or over-smoothed
prediction reduces the residual while losing the solution. The accuracy
metric is therefore always reported next to the residual, and in this
study the physics model *improved* accuracy as well (0.081 -> 0.037 on
Burgers), so the improvement is not a smoothing artefact.

---

## D8 - A separate resolution-aware restriction is used, never a stride

The reference grid is twice the observation grid in each direction. A
naive stride would alias. The restriction removes the modes above the
observation band-limit with a symmetric `|k|` mask on the full FFT
before sampling. See `docs/DATA.md`.

---

## D9 - The study runs one seed at a time and the confirmatory matrix is not executed automatically

`experiment_concurrency = 1` and a project job lock enforce sequential
execution. The full 18-run matrix is **not** run by default
(`run_full_sweep_by_default: false`); it is prepared, costed and
available behind an explicit command. The `full` profile must be
requested explicitly.

The prototype matrix is one seed per condition. The three-seed
requirement applies to independent model *training*, not to three
inference repetitions of one checkpoint, and it is **not yet met** -
see `docs/STATUS.md`.

---

## D10 - A near parameter-matched comparison is declared, with its limits

The FNO and the U-Net are not parameter matched at the pilot widths:

| model | parameters |
|---|---|
| burgers FNO1D (width 16, 16 modes, 4 blocks) | 19,825 (16,384 complex) |
| burgers U-Net1D (width 16, 3 levels) | 191,329 |
| allen_cahn FNO2D (width 16, 8x8 modes, 4 blocks) | 68,993 (65,536 complex) |
| allen_cahn U-Net2D (width 16, 3 levels) | 558,529 |

The U-Net is roughly 8-10x larger, so the U-Net's *worse* accuracy in
this study cannot be attributed to a parameter disadvantage; the
comparison is, if anything, generous to the U-Net.

**Not done:** parameter matching is not claimed to equalise FLOPs or
optimisation difficulty, and a near-matched U-Net variant is listed as
a pending ablation rather than presented as a completed comparison.
