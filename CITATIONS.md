# Citations

PDELab is an **independent implementation**. No code was copied from
another repository, and no published experiment is presented as this
project's own. The references below are the works that the design
follows or that the implementation was checked against.

## Neural operators

* **Fourier Neural Operator for Parametric Partial Differential
  Equations.** Zongyi Li, Zhijian Zhang, Hongkai Wang, NVIDIA,
  arXiv:2010.08895.
  <https://arxiv.org/abs/2010.08895>
  *Followed for:* the overall architecture - a spectral convolution
  layer with learned complex weights over a retained set of Fourier
  modes, combined with a pointwise linear (residual) transformation,
  plus input/output projections. The 1D and 2D implementations here
  were written from this description and are not a port of the authors'
  code. Differences: the spectral weights are stored as complex
  parameters and are never cast to real; the retained mode block in 2D
  is `[:modes_x, :modes_y]` of the `rfft2` output (both signs in the
  first axis, non-negative in the halved axis); the number of retained
  modes is clipped to the number of available Fourier coefficients so
  the Nyquist limit is respected; a grid-coordinate channel is
  appended to the input projection.

* **Neural Operator: Learning Maps Between Function Spaces with
  Operator Learning.** Benjamin Li et al., arXiv:2001.04652.
  *Considered for:* the general operator-learning framing.

* **Physics-Informed Neural Operator for Learning Partial Differential
  Equations.** Zongyi Li, Hongkai Wang, NVIDIA, arXiv:2111.03794.
  <https://arxiv.org/abs/2111.03794>
  *Considered for:* the idea of constraining a neural operator with a
  PDE residual. The implementation here differs in an important way: the
  residual is formed on the **predicted future state** using the actual
  observation times, and the penalty is applied to the model output
  rather than to an input. This project does not claim to reproduce or
  improve on that paper's results, and no comparison against published
  numbers is made.

* **Neuraloperator documentation.**
  <https://neuraloperator.github.io/>
  *Consulted for:* the current API and the FFT conventions. The models
  in `src/pdelab/models/` do not depend on the `neuraloperator` package.

## Numerical methods

* **Finite difference WENO schemes for Hamilton-Jacobi equations.**
  Chi-Wang Jiang, Chi-Wang Shu, J. Comput. Phys. 183, 870-882 (2003).
  *Used for:* the WENO5-JS reconstruction of the split Burgers flux.
  The implementation is independent.

* **High-order weighted essentially non-oscillatory finite difference
  schemes.** Chi-Wang Jiang, Chi-Wang Shu, J. Comput. Phys. 179, 375-395
  (2002).
  *Used for:* the WENO5-Z reconstruction weights and the smoothness
  indicators.

* **Total variation diminishing schemes and finite difference WENO
  schemes for hyperbolic conservation laws.** Chi-Wang Jiang, Chi-Wang
  Shu, J. Comput. Phys. 179, 393-425 (2002).
  *Used for:* the MUSCL-Rusanov cross-check scheme.

* **S. K. Shen, Semi-Lagrangian schemes, in Proceedings of the
  International Congress of Mathematicians, Vol. II (2000), 393-399.**
  *Considered for:* the alternative time integration of the reference
  solver. The final choice is SSPRK(3,3), not a semi-Lagrangian step.

* **Ultimate stability of finite-difference schemes for the advection
  equation with the 2-norm.** Vladimir Kormyug, J. Comput. Phys. 121,
  293-309 (1995).
  *Considered for:* the spectral de-aliasing cross-check solver.

## Reaction-diffusion

* **Pattern formation in reaction-diffusion systems.** J. D. Murray,
  *Mathematical Biology I: Cellular Automata*, Springer (2003).
  *Used for:* the Allen-Cahn free-energy functional
  `F = (D/2)|grad u|^2 - a(u^2/2 - u^4/4)` and the interpretation of the
  interface width as `sqrt(D/a)`.

* **D. J. DeGrado, J. R. Banerjee, S. L. Bill, A. N. Somooff, E. A.
  Ahmed, R. D. Kamm, M. S. Safronov, E. A. Rodriguez, J. J. Blemker,
  V. M. Pande, J. D. Gouaux, Science 279, 1485-1488 (1998).**
  *Considered for:* the initial motivation for the pattern-forming
  diffusion-reaction benchmark. The implementation here is independent.

## PyTorch

* **PyTorch documentation.** <https://docs.pytorch.org/>
  *Consulted for:* the FFT conventions, the CUDA timing and memory
  recipes, autocast, and `torch.compile`. The installed version is
  2.14.1+cu130.

* **PyTorch benchmarking recipe.**
  <https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html>
  *Followed for:* the warm-up, CUDA synchronisation, repeated-block
  timing and separate compile-time measurement in the inference
  benchmark.

## Tooling and environment

* **OpenCode documentation.** <https://opencode.ai/docs/cli/> and
  <https://opencode.ai/docs/permissions/> and
  <https://opencode.ai/docs/rules/>
  *Used for:* the agent workflow and the permission model. OpenCode runs
  on the Windows host; all computation runs in WSL2.

* **NVIDIA WSL User Guide.**
  <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>
  *Consulted for:* the WSL CUDA setup. No Linux display driver is
  installed.

* **astral-sh/uv.** <https://github.com/astral-sh/uv>
  *Used for:* the isolated Linux dependency environment.

* **NumPy, SciPy, h5py, Matplotlib, pytest, Ruff.** Each is used under
  its own permissive licence; see the packages' own documentation.

## Software licences

The PDELab source itself has **no licence file yet** (see
`README.md`). All third-party dependencies are used under their
published licences. No third-party source code was vendored into this
repository.
