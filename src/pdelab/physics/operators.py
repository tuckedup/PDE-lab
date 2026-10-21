"""Validated periodic differential operators (differentiable).

All operators are spectral and periodic, so they are consistent with the
periodic benchmark domains used by the solvers.

De-aliasing
-----------
The nonlinear products are evaluated on an explicit **retained band**:

* the input field is projected onto the retained band,
* the product is formed in physical space, and
* the **output** is projected back onto the same retained band.

Projecting only the input is *not* sufficient: the product of a field
with modes ``|k| <= K`` reaches ``2K`` (quadratic) or ``3K`` (cubic),
which is above the grid Nyquist ``n / 2`` for the classical 2/3 and 1/2
rules alike, so the excess folds back into the retained band.  The
masks are Hermitian symmetric (``mask[k] == mask[n - k]``) and exclude
the Nyquist mode explicitly, so a real field stays real.  The retained
bands are

* Burgers (quadratic product): ``|k| <= n // 3`` (the 2/3 rule),
* Allen-Cahn (cubic product): ``|k| <= n // 4`` (the 1/2 rule).

The same policies are used by the reference solvers, so the residual of
the reference solution measured with these operators is the residual of
the *same* discrete operator.

Time derivatives
----------------
Two different time stencils are provided and they must not be confused:

``time_derivative_forward_first_order``
    The forward difference ``(u_next - u_prev) / dt``.  Together with
    the spatial operator evaluated on ``u_next`` this is the
    **implicit-Euler form** of the residual (first order in time).

``time_derivative_centered_first_order``
    The centred difference ``(u_next - u_prev) / (2 dt)``.  It needs
    the snapshot *before* and the snapshot *after* the centre, so its
    support is the **interior snapshots only** (indices ``1 .. T-2`` of
    a trajectory with ``T`` snapshots).  The residual built from it is
    second order in time and is evaluated on the *centre* snapshot.

Both residuals are provided for each system with names that state the
time stencil explicitly:

* ``burgers_residual_implicit_euler`` / ``allen_cahn_residual_implicit_euler``
* ``burgers_residual_centered_first_derivative`` /
  ``allen_cahn_residual_centered_first_derivative``

The two evaluators share the same spatial operator, so they differ only
in their time stencil.  Their supports differ: the centred stencil has
``T - 2`` interior entries, the forward stencil has ``T - 1``; when the
two are compared, the forward one must be cut to the shared interior
support.
"""

from __future__ import annotations

import numpy as np
import torch


def spectral_k_1d(n: int, L: float = 1.0, device=None, dtype=torch.float32) -> torch.Tensor:
    """Wavenumbers for a periodic domain of length L sampled at n points."""
    dx = L / n
    k = 2.0 * np.pi * torch.fft.fftfreq(n, d=float(dx))
    return k.to(device=device, dtype=dtype)


def spectral_k_2d(
    nx: int, ny: int, L: float = 1.0, device=None, dtype=torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    dx = L / nx
    dy = L / ny
    kx = 2.0 * np.pi * torch.fft.fftfreq(nx, d=float(dx))
    ky = 2.0 * np.pi * torch.fft.fftfreq(ny, d=float(dy))
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    return KX.to(device=device, dtype=dtype), KY.to(device=device, dtype=dtype)


def _mode_numbers(n: int) -> np.ndarray:
    """Signed mode numbers of a full FFT of length ``n`` (FFT ordering)."""
    return np.rint(np.fft.fftfreq(n) * n).astype(int)


def dealias_2_3(n: int) -> np.ndarray:
    """Boolean mask implementing the 2/3 rule on an FFT of length n.

    The mask keeps the modes ``|k| <= n // 3`` of the *full* FFT
    ordering.  It is Hermitian symmetric (``mask[k] == mask[n - k]``)
    and the Nyquist mode (its own mirror) is excluded, so a real field
    stays real.  This is the classical alias-free band for a
    *quadratic* product.
    """
    if n < 4:
        raise ValueError(f"the 2/3 rule needs at least 4 points, got {n}")
    return np.abs(_mode_numbers(n)) <= (n // 3)


def dealias_1_2(n: int) -> np.ndarray:
    """Boolean mask implementing the 1/2 rule on an FFT of length n.

    The mask keeps the modes ``|k| <= n // 4`` of the *full* FFT
    ordering.  It is Hermitian symmetric and excludes the Nyquist mode.
    This is the classical alias-free band for a *cubic* product: the
    2/3 rule is not sufficient there, because the cubic product reaches
    ``3 * n // 3 = n`` and folds back into the retained band.
    """
    if n < 4:
        raise ValueError(f"the 1/2 rule needs at least 4 points, got {n}")
    return np.abs(_mode_numbers(n)) <= (n // 4)


def ddx_1d(u: torch.Tensor, k: torch.Tensor | None = None) -> torch.Tensor:
    """First periodic derivative along the last axis. u: (..., X) -> (..., X)."""
    n = u.shape[-1]
    if k is None:
        k = spectral_k_1d(n, device=u.device, dtype=u.dtype)
    u_hat = torch.fft.fft(u, dim=-1)
    d_hat = 1j * k.to(u.device, u.dtype) * u_hat
    return torch.fft.ifft(d_hat, dim=-1).real


def d2x_1d(u: torch.Tensor, k: torch.Tensor | None = None) -> torch.Tensor:
    """Second periodic derivative along the last axis."""
    n = u.shape[-1]
    if k is None:
        k = spectral_k_1d(n, device=u.device, dtype=u.dtype)
    u_hat = torch.fft.fft(u, dim=-1)
    d_hat = -(k.to(u.device, u.dtype) ** 2) * u_hat
    return torch.fft.ifft(d_hat, dim=-1).real


def laplacian_2d(
    u: torch.Tensor, kx: torch.Tensor | None = None, ky: torch.Tensor | None = None
) -> torch.Tensor:
    """Periodic Laplacian for u: (..., X, Y) -> (..., X, Y)."""
    nx, ny = u.shape[-2], u.shape[-1]
    if kx is None or ky is None:
        kx, ky = spectral_k_2d(nx, ny, device=u.device, dtype=u.dtype)
    u_hat = torch.fft.fft2(u, dim=(-2, -1))
    lap_hat = -(kx.to(u.device, u.dtype) ** 2 + ky.to(u.device, u.dtype) ** 2) * u_hat
    return torch.fft.ifft2(lap_hat, dim=(-2, -1)).real


def grad_x_2d(
    u: torch.Tensor, kx: torch.Tensor | None = None, ky: torch.Tensor | None = None
) -> torch.Tensor:
    """Periodic d/dx for u: (..., X, Y) -> (..., X, Y)."""
    nx, ny = u.shape[-2], u.shape[-1]
    if kx is None or ky is None:
        kx, ky = spectral_k_2d(nx, ny, device=u.device, dtype=u.dtype)
    u_hat = torch.fft.fft2(u, dim=(-2, -1))
    d_hat = 1j * kx.to(u.device, u.dtype) * u_hat
    return torch.fft.ifft2(d_hat, dim=(-2, -1)).real


# ----------------------------------------------------------------------
# time derivatives
# ----------------------------------------------------------------------
def time_derivative_forward_first_order(
    u_next: torch.Tensor, u_prev: torch.Tensor, dt: float
) -> torch.Tensor:
    """Forward-difference time derivative between two physical snapshots.

    This is the time derivative implied by the observation sampling when
    the model predicts ``u_next`` from ``u_prev`` across an interval
    ``dt``.  It is a genuine forward-time difference of the (predicted)
    trajectory and is used to close the PDE residual on predicted states.
    Together with the spatial operator evaluated on ``u_next`` it is the
    **implicit-Euler form** of the residual (first order in time).
    """
    return (u_next - u_prev) / dt


def time_derivative_centered_first_order(
    u_next: torch.Tensor, u_prev: torch.Tensor, dt: float
) -> torch.Tensor:
    """Centred first time derivative on the *interior* of a trajectory.

    ``u_next`` is the snapshot at ``t + dt`` and ``u_prev`` the snapshot
    at ``t - dt``; the result is the derivative at the centre ``t``:

        (u_next - u_prev) / (2 dt)

    **Support:** the interior snapshots only.  For a trajectory with
    ``T`` snapshots the result has ``T - 2`` entries, indexed by the
    centre times ``t_1 .. t_{T-2}``.  This is second order in time.

    Do not confuse this with
    :func:`time_derivative_forward_first_order`, which is the forward
    difference used by the implicit-Euler-form residual.
    """
    return (u_next - u_prev) / (2.0 * dt)


def time_derivative_autograd(
    u_next: torch.Tensor, u_prev: torch.Tensor, dt: float
) -> torch.Tensor:
    """Legacy alias of :func:`time_derivative_forward_first_order`.

    The name is historical: no autograd machinery is involved, the
    operator is a plain forward difference.
    """
    return time_derivative_forward_first_order(u_next, u_prev, dt)


# ----------------------------------------------------------------------
# retained-band products
# ----------------------------------------------------------------------
def burgers_flux_dealiased(u: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Retained-band Burgers flux ``u^2 / 2`` (1D).

    The input is projected onto the retained band, the product is formed
    in physical space and the output is projected back onto the same
    band.  ``u`` has shape (..., X); the result has the same shape.
    """
    n = u.shape[-1]
    if mask is None:
        mask = torch.as_tensor(dealias_2_3(n), device=u.device, dtype=torch.bool)
    u_filt = torch.fft.ifft(torch.fft.fft(u, dim=-1) * mask, dim=-1).real
    flux_hat = torch.fft.fft(0.5 * u_filt**2, dim=-1) * mask
    return torch.fft.ifft(flux_hat, dim=-1).real


def allen_cahn_cubic_dealiased(
    u: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Retained-band cubic ``u^3`` (2D).

    The input is projected onto the retained band, the cube is formed in
    physical space and the output is projected back onto the same band.
    ``u`` has shape (..., X, Y); the result has the same shape.
    """
    nx = u.shape[-2]
    if mask is None:
        mask = torch.as_tensor(dealias_1_2(nx), device=u.device, dtype=torch.bool)
        mask = mask[:, None] & mask[None, :]
    u_filt = torch.fft.ifft2(torch.fft.fft2(u) * mask).real
    cubic_hat = torch.fft.fft2(u_filt**3) * mask
    return torch.fft.ifft2(cubic_hat).real


# ----------------------------------------------------------------------
# Burgers residuals
# ----------------------------------------------------------------------
def burgers_residual_implicit_euler(
    u_next: torch.Tensor,
    u_prev: torch.Tensor,
    dt: float,
    nu: float,
    dealias: bool = True,
) -> torch.Tensor:
    """Burgers residual in the implicit-Euler form (first order in time).

    R = (u_next - u_prev) / dt + (u^2/2)_x - nu * u_xx

    The time derivative is the **forward** difference between the two
    given snapshots and the spatial operator is evaluated on
    ``u_next``; this is exactly the backward-Euler / implicit-Euler
    discretisation of ``u_t = -(u^2/2)_x + nu u_xx`` at time ``t + dt``.

    Parameters
    ----------
    u_next:
        State at t + dt (may be a prediction), shape (B, X).
    u_prev:
        State at t (may be ground truth, detached), shape (B, X).
    dt:
        Observation time step.
    nu:
        Viscosity.
    dealias:
        Evaluate the flux product on the retained band (input filter and
        output projection).

    Returns
    -------
    Residual tensor of shape (B, X).
    """
    u_t = time_derivative_forward_first_order(u_next, u_prev, dt)
    n = u_next.shape[-1]
    k = spectral_k_1d(n, device=u_next.device, dtype=u_next.dtype)
    if dealias:
        flux = burgers_flux_dealiased(u_next)
    else:
        flux = 0.5 * u_next**2
    flux_x = torch.fft.ifft(1j * k * torch.fft.fft(flux, dim=-1), dim=-1).real
    u_xx = d2x_1d(u_next, k)
    return u_t + flux_x - nu * u_xx


def burgers_residual_centered_first_derivative(
    traj: torch.Tensor,
    dt: float,
    nu: float,
    dealias: bool = True,
) -> torch.Tensor:
    """Burgers residual with the **centred first** time derivative.

    R = (u_{i+1} - u_{i-1}) / (2 dt) + (u^2/2)_x - nu * u_xx

    with the spatial operator evaluated on the **centre** snapshot
    ``u_i``.

    **Support:** the interior snapshots only.  For a trajectory with
    ``T`` snapshots the result has shape ``(B, T - 2, X)`` and entry
    ``j`` corresponds to the centre time ``t_{j+1}``.

    This operator is second order in time; the implicit-Euler-form
    residual :func:`burgers_residual_implicit_euler` is first order.
    """
    if traj.dim() < 3:
        raise ValueError(
            "the centred first derivative needs a batch, a time and a spatial axis"
        )
    if traj.shape[1] < 3:
        raise ValueError(
            f"the interior support needs at least three snapshots, got {traj.shape[1]}"
        )
    u_t = time_derivative_centered_first_order(traj[:, 2:], traj[:, :-2], dt)
    centre = traj[:, 1:-1]
    n = centre.shape[-1]
    k = spectral_k_1d(n, device=centre.device, dtype=centre.dtype)
    if dealias:
        flux = burgers_flux_dealiased(centre)
    else:
        flux = 0.5 * centre**2
    flux_x = torch.fft.ifft(1j * k * torch.fft.fft(flux, dim=-1), dim=-1).real
    u_xx = d2x_1d(centre, k)
    return u_t + flux_x - nu * u_xx


def burgers_residual(
    u_next: torch.Tensor,
    u_prev: torch.Tensor,
    dt: float,
    nu: float,
    dealias: bool = True,
) -> torch.Tensor:
    """Alias of :func:`burgers_residual_implicit_euler`.

    Kept for backwards compatibility; the explicit name is preferred
    because it states the time stencil.
    """
    return burgers_residual_implicit_euler(u_next, u_prev, dt, nu, dealias=dealias)


# ----------------------------------------------------------------------
# Allen-Cahn residuals
# ----------------------------------------------------------------------
def allen_cahn_residual_implicit_euler(
    u_next: torch.Tensor,
    u_prev: torch.Tensor,
    dt: float,
    D: float,
    a: float,
    dealias: bool = True,
) -> torch.Tensor:
    """Allen-Cahn residual in the implicit-Euler form (first order in time).

    R = (u_next - u_prev) / dt - D * laplacian(u) - a * (u - u^3)

    The time derivative is the **forward** difference between the two
    given snapshots and the spatial operator is evaluated on
    ``u_next``.

    Shapes: u_next / u_prev are (B, X, Y); result is (B, X, Y).
    """
    u_t = time_derivative_forward_first_order(u_next, u_prev, dt)
    nx, ny = u_next.shape[-2], u_next.shape[-1]
    kx, ky = spectral_k_2d(nx, ny, device=u_next.device, dtype=u_next.dtype)
    lap = laplacian_2d(u_next, kx, ky)
    if dealias:
        cubic = allen_cahn_cubic_dealiased(u_next)
    else:
        cubic = u_next**3
    return u_t - D * lap - a * (u_next - cubic)


def allen_cahn_residual_centered_first_derivative(
    traj: torch.Tensor,
    dt: float,
    D: float,
    a: float,
    dealias: bool = True,
) -> torch.Tensor:
    """Allen-Cahn residual with the **centred first** time derivative.

    R = (u_{i+1} - u_{i-1}) / (2 dt) - D * laplacian(u_i) - a * (u_i - u_i^3)

    with the spatial operator evaluated on the **centre** snapshot.

    **Support:** the interior snapshots only.  For a trajectory with
    ``T`` snapshots the result has shape ``(B, T - 2, X, Y)`` and entry
    ``j`` corresponds to the centre time ``t_{j+1}``.

    This operator is second order in time; the implicit-Euler-form
    residual :func:`allen_cahn_residual_implicit_euler` is first order.
    """
    if traj.dim() < 4:
        raise ValueError(
            "the centred first derivative needs a batch, a time and two spatial axes"
        )
    if traj.shape[1] < 3:
        raise ValueError(
            f"the interior support needs at least three snapshots, got {traj.shape[1]}"
        )
    u_t = time_derivative_centered_first_order(traj[:, 2:], traj[:, :-2], dt)
    centre = traj[:, 1:-1]
    nx, ny = centre.shape[-2], centre.shape[-1]
    kx, ky = spectral_k_2d(nx, ny, device=centre.device, dtype=centre.dtype)
    lap = laplacian_2d(centre, kx, ky)
    if dealias:
        cubic = allen_cahn_cubic_dealiased(centre)
    else:
        cubic = centre**3
    return u_t - D * lap - a * (centre - cubic)


def allen_cahn_residual(
    u_next: torch.Tensor,
    u_prev: torch.Tensor,
    dt: float,
    D: float,
    a: float,
    dealias: bool = True,
) -> torch.Tensor:
    """Alias of :func:`allen_cahn_residual_implicit_euler`.

    Kept for backwards compatibility; the explicit name is preferred
    because it states the time stencil.
    """
    return allen_cahn_residual_implicit_euler(u_next, u_prev, dt, D, a, dealias=dealias)


def energy_1d(u: torch.Tensor) -> torch.Tensor:
    """Integral of u^2/2 over a unit periodic domain (per sample)."""
    n = u.shape[-1]
    return 0.5 * torch.sum(u**2, dim=-1) / n


def mass_1d(u: torch.Tensor) -> torch.Tensor:
    """Mean of u over a unit periodic domain (per sample)."""
    return torch.mean(u, dim=-1)


def mass_2d(u: torch.Tensor) -> torch.Tensor:
    """Mean of u over a unit periodic square (per sample)."""
    return torch.mean(u, dim=(-2, -1))
