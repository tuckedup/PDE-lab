# Reference solvers

All checks are quantitative with declared tolerances and are executed by

```bash
.venv/bin/python scripts/linux/validate_solvers.py     # or
.venv/bin/python -m pdelab.cli solvers validate --system burgers
.venv/bin/python -m pdelab.cli solvers validate --system allen_cahn
```

**Current status: 12/12 checks pass for Burgers, 12/12 for Allen-Cahn.**

---

## Burgers - WENO5-JS finite volume (primary generator)

| ingredient | choice |
|---|---|
| scheme | conservative finite volume, WENO5 (Jiang-Shu) reconstruction of the split fluxes |
| flux | `f(u) = u^2/2`, split into `f+/f-`, reconstructed with 3rd-order optimal weights and the Jiang-Shu smoothness indicators |
| diffusion | 2nd-order central difference |
| time integration | SSPRK(3,3) (Shu-Osher) |
| stability policy | `dt <= cfl * dx / (2 max|u|)` and `dt <= 0.5 dx^2 / nu`, with `cfl = 0.4` |
| boundary | fully periodic (`np.roll`) |
| grid | 256 reference cells, 128 observation cells |

### Why WENO5 and not a TVD scheme

The first implementation used a minmod-limited MUSCL-Rusanov scheme. The
validation suite showed its *global* L2 convergence collapsing to
**first order** (measured order 1.00 at every refinement), and a
measured 2.5% error against a converged spectral solution at n = 1024.
That is far too large for a reference.

Two independent bugs were found while diagnosing this, and both are
worth recording because a self-convergence test could not have found
either:

1. **The flux divergence was mis-indexed.** `flux[i]` is the flux at
   the face `x_{i+1/2}`, so the divergence at cell centre `x_i` is
   `(F_{i+1/2} - F_{i-1/2})/dx`. The original code used
   `(np.roll(flux, -1) - flux)/dx`, which is centred at `x_{i+1}`: a
   **full-cell shift**. The scheme stayed consistent, so it still
   "converged" - just at first order, to the wrong answer.

2. **The SSPRK(3,3) stage arguments and combination weights were
   mismatched.** The original form `k2 = f(u + dt/2 k1)`,
   `k3 = f(u + dt/2 k2)`, `u += dt/6 (k1 + 2 k2 + 2 k3)` is not
   consistent: applied to `du/dt = 0.7 u` it advances by `5/6` of the
   exact rate. The verified form is
   `u1 = u + dt k1`, `u2 = 3/4 u + 1/4 (u1 + dt k2)`,
   `u3 = 1/3 u + 2/3 (u2 + dt k3)`, which reproduces
   `exp(0.7 t)` to 2e-11 over 1000 steps.

With both fixed, the measured convergence is 1.93-2.00 (smooth regime)
and the error against a converged spectral solution is 3.9e-5 at
n = 256 in the shock-forming regime - about 12x smaller than the
MUSCL scheme's 4.2e-4.

WENO5 is used rather than the TVD scheme because it is also markedly
more accurate once a shock forms, which is the regime the benchmark
actually lives in.

### Cross-checks

Two further independent discretizations are implemented:

* `Burgers1DFiniteVolume` - MUSCL-Rusanov (kept as a cross-check).
* `Burgers1DSpectral` - dealiased pseudo-spectral with Strang splitting
  and an exact integrating-factor diffusion step.

All three agree to a relative L2 of 9.4e-6 (WENO5 vs spectral) and
4.4e-4 (WENO5 vs MUSCL) at t = 0.02, n = 128, while the solution is
still smooth.

### Validation checks

| check | result |
|---|---|
| constant state is stationary | max abs error 0.0 |
| time-integrator consistency (`du/dt = 0.7 u`) | 2.0e-11 |
| spatial operator vs the exact continuous RHS | order 2.52, 2.01 |
| diffusion operator vs the analytic decay of one mode | 3.0e-5 |
| mean conservation over a shock-forming run | 1.6e-16 |
| viscous energy is non-increasing | max increase -2.1e-4 |
| spatial refinement, smooth regime (t = 0.02) | order 1.71, 1.55 |
| spatial refinement, shock regime (t = 0.1) | order 1.28, 1.39 |
| temporal refinement at fixed space | order 1.59, 2.72 |
| agreement of three independent discretizations | 9.4e-6 / 4.4e-4 |
| manufactured solution with a recorded forcing | 4.8e-9 |
| **reference-solution own error** (n=256 vs n=512) | **2.2e-4 relative L2** |

The last row is the important one for interpreting the learning
results: the reference is an *approximation* of the true solution, with
a measured own error of 2.2e-4, not exact ground truth.

### Restriction to the observation grid

The reference is computed on 256 cells and restricted to 128. The
restriction is **not** a naive stride. Modes above the observation
band-limit are removed with a symmetric `|k|` mask on the full FFT and
the field is then sampled. A naive stride would alias the modes between
the observation Nyquist and the reference Nyquist onto the observation
grid.

---

## Allen-Cahn - pseudo-spectral with Strang splitting

| ingredient | choice |
|---|---|
| space | pseudo-spectral; the Laplacian is exact in Fourier space |
| nonlinearity | `a (u - u^3)` evaluated on the 2/3-rule filtered field |
| diffusion | **exact** Fourier exponential |
| time integration | Strang splitting: half reaction (RK4), full diffusion, half reaction |
| stability | no explicit diffusion restriction; the reaction RK4 limit is `dt <= 2.8/(3a)` |
| accuracy | the reference step is `dt <= 0.005/a`, chosen from the measured temporal-refinement study |
| boundary | fully periodic |

### Validation checks

| check | result |
|---|---|
| equilibria u = 0, +1, -1 are stationary | max abs error 0.0 |
| reaction law against the exact quartic solution | 5.6e-16 |
| linear diffusion rate `exp(-D k^2 t)` | 3.3e-18 |
| free energy is non-increasing | -0.0266 -> -0.1259 |
| spatial refinement | order 11.5, 19.9 (spectral) |
| temporal refinement | order 2.07, 2.32 |
| mass is **not** conserved (guard) | drift 0.268 |
| manufactured solution with a recorded forcing | 4.8e-9 |
| **reference-solution own error** (n=64 vs n=128) | **9.6e-8 relative L2** |

### A diagnostic bug worth recording

The first energy helper used a `1/2` coefficient on the gradient term
instead of `D/2`, and consequently reported a free energy that
*increased* from 2.44 to 22.43 during a run. This looked exactly like a
solver instability, and the natural (but wrong) conclusion was that the
cubic nonlinearity was aliasing into the retained band.

Two hypotheses were tested before the real cause was found:

* *Aliasing?* The 2/3 rule is alias-free for a **quadratic** product
  but not for a **cubic** one, so the reaction band was narrowed to
  `N/6`. The result was **bit-for-bit identical** at `N/3` and `N/6`,
  which ruled aliasing out.
* *Under-resolution?* The result was identical at n = 32, 64, 128 and
  256, which ruled grid-dependence out.

The actual cause was the energy functional itself: the free energy of
`u_t = D laplacian(u) + a(u - u^3)` is
`F = (D/2)|grad u|^2 - a(u^2/2 - u^4/4)`, because
`-delta F/delta u = D laplacian(u) + a(u - u^3)`. With the correct
coefficient, `dF/dt = -<u_t^2> <= 0` and the free energy decreases, as
theory requires.

## Manufactured solutions

Both systems have a manufactured-solution check in which a forcing term
is **recorded** and applied to make the discrete equation exact at an
analytic field. The forcing is a test artefact only: it is **excluded**
from the unforced main benchmark, and no manufactured data enters any
learning run.
