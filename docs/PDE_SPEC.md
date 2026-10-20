# PDE specifications

Both systems are nondimensional. Length is scaled so the domain has unit
extent, and time is scaled so that the characteristic advection /
reaction speed is O(1). The domains are **periodic** so that the Fourier
and spatial baselines share a consistent benchmark; other boundary
conditions are a later extension, not a hidden architecture advantage.

---

## System A - 1D viscous Burgers

```
u_t + (u^2 / 2)_x = nu * u_xx
```

| quantity | value |
|---|---|
| domain | periodic x in [0, 1), endpoint NOT duplicated |
| viscosity `nu` | 0.01 (fixed for the main benchmark) |
| length scale | L = 1 |
| time scale | chosen so the characteristic advection speed is O(1) |
| reference grid | 256 cells |
| observation grid | 128 cells |
| observation spacing | dt = 0.005 |
| horizon | t in [0, 0.5] -> 101 snapshots, 100 rollout steps |

### Units and scales

`u` is nondimensional. The viscous length scale is `nu / u_max ~ 0.01`,
so the shock layer that forms during the run is about 10-20 cells wide
on the reference grid and is therefore resolved.

### Initial conditions

Smooth, band-limited random Fourier series with a recorded spectrum:

```
u0(x) = sum_{k=1..K} [ a_k cos(2 pi k x) + b_k sin(2 pi k x) ]
a_k, b_k ~ N(0, 1),  K = 6,  amplitude scale 0.9
```

The realised RMS, the realised maximum and every coefficient are written
into the dataset manifest, so the sampler can be audited without
re-running it.

### Conservation and diagnostics

* The spatial mean is **exactly** conserved by the periodic equation
  (the divergence term integrates to zero). Measured drift over a
  shock-forming run: 1.6e-16.
* The viscous energy `0.5 * <u^2>` is non-increasing. Measured maximum
  increase: -2.1e-4 (i.e. a decrease).
* Mass conservation is a property of this system and **not** of
  System B; see below.

---

## System B - 2D diffusion-reaction (Allen-Cahn)

```
u_t = D * (u_xx + u_yy) + a * (u - u^3)
```

This is explicitly the **scalar Allen-Cahn** system. It is **not**
Gray-Scott and no two-species dataset is involved. No result from a
different PDE is ever reported in the same table row.

| quantity | value |
|---|---|
| domain | periodic unit square [0,1) x [0,1), endpoints NOT duplicated |
| diffusion `D` | 0.001 |
| reaction rate `a` | 1.0 |
| reference grid | 64 x 64 |
| observation grid | 32 x 32 |
| observation spacing | dt = 0.1 |
| horizon | t in [0, 3.0] -> 31 snapshots, 30 rollout steps |

`D = 0.001` and `a = 1` are **starting configuration choices**, not
published benchmark constants. They were selected on measured evidence
(`scripts/linux/study_ac_params.py`):

* the 32x32 observation grid and the 64x64 reference agree to a relative
  L2 difference below 1e-3, so the reference is grid-converged at the
  observation resolution;
* the persistence baseline over the horizon is 0.41, i.e. more than
  twice the declared 0.20 accuracy target, so the task is not trivial;
* reference generation costs 0.01 s per trajectory.

### Interface width

The Allen-Cahn interface width scales as `sqrt(D / a) = 0.032`, which is
about one cell at the 32x32 observation resolution. This is acceptable
here because the initial conditions are band-limited (modes up to 3) and
the measured grid sensitivity of the observation grid against the
reference grid is below 1e-3.

### Conservation and diagnostics

* Mass is **not** conserved for this system. The reaction term
  `a(u - u^3)` has a non-zero spatial integral in general. A
  validation check asserts that the mass drift is *non-negligible*, so
  that no false conservation constraint is ever applied to it. Measured
  drift with an offset initial condition: 0.27.
* The free energy of the equation is

  ```
  F[u] = integral of [ (D/2) |grad u|^2 - a ( u^2/2 - u^4/4 ) ]
  ```

  because `-delta F / delta u = D laplacian(u) + a (u - u^3)`, which is
  exactly the equation being integrated. Along a solution
  `dF/dt = -<u_t^2> <= 0`, so `F` must be non-increasing; this is
  asserted by a validation check. Measured: -0.0266 -> -0.1259.

  The `D/2` coefficient on the gradient term is essential. An earlier
  version of the energy helper used `1/2` and consequently reported an
  *increasing* free energy, which looked like a solver instability but
  was a bug in the diagnostic.

---

## Why these horizons

The brief requires a meaningful prediction horizon: copying the input
must not nearly meet the accuracy goal. The measured persistence
baseline (the nRMSE of copying the initial frame over the whole
horizon) is:

| system | horizon | observation dt | persistence nRMSE | declared target |
|---|---|---|---|---|
| burgers | T = 0.5 | 0.005 | **1.478** | 0.05 |
| allen_cahn (first pilot) | T = 1.0 | 0.05 | 0.197 | 0.20 |
| allen_cahn (revised) | T = 3.0 | 0.1 | **0.415** | 0.20 |

The 2D horizon was **revised after the first pilot** because the
persistence baseline of 0.197 almost met the declared 0.20 target, i.e.
the trivial baseline nearly solved the task. The revision extends the
horizon threefold, which makes the benchmark **harder**; it was not made
to move a result in either direction. See `docs/DECISIONS.md`.

## Residual definitions

Residuals are evaluated in physical (nondimensional) units:

```
R_burgers     = u_t + (u^2/2)_x - nu * u_xx
R_allen_cahn  = u_t - D * laplacian(u) - a * (u - u^3)
```

The time derivative is the forward difference between consecutive
observation snapshots, i.e. the derivative implied by the actual
observation times. Spatial derivatives use validated periodic spectral
operators; the Burgers nonlinear flux uses the 2/3 de-aliasing rule.

The **primary** residual evaluator therefore has an O(dt * u_tt)
truncation error, so even an exact PDE solution sampled at `dt` shows a
non-zero residual. That floor is measured and reported: it is 1.08 for
Burgers and 0.061 for Allen-Cahn. A **separately declared** second-order
(centred) evaluator is also reported as a diagnostic, under a distinct
name, and is never mixed with the primary one. This makes it possible to
see whether a residual improvement is specific to one operator.
