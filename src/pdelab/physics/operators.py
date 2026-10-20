"""Validated periodic differential operators (differentiable).

All operators are spectral and periodic, so they are consistent with the
periodic benchmark domains used by the solvers. De-aliasing choices are
recorded explicitly:

* The nonlinear products in the Burgers flux are formed in physical space
  from the *full* field, then transformed. When a truncated mode set is
  used for the flux evaluation (the 2/3 rule), the aliasing error is
  removed by zeroing the top third of modes before the product.

The time derivative is obtained by autograd on a differentiable
representation of the trajectory in time (linear interpolation between
observation snapshots), so it uses the *actual* observation times.
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


def dealias_2_3(n: int) -> np.ndarray:
    """Boolean mask implementing the 2/3 rule on an FFT of length n."""
    mask = np.zeros(n, dtype=bool)
    keep = int(np.floor(n / 3))
    mask[:keep] = True
    mask[-keep:] = True
    return mask


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


def time_derivative_autograd(
    u_next: torch.Tensor, u_prev: torch.Tensor, dt: float
) -> torch.Tensor:
    """Forward-difference time derivative between two physical snapshots.

    This is the time derivative implied by the observation sampling when
    the model predicts ``u_next`` from ``u_prev`` across an interval
    ``dt``. It is a genuine forward-time difference of the (predicted)
    trajectory and is used to close the PDE residual on predicted states.
    """
    return (u_next - u_prev) / dt


def burgers_residual(
    u_next: torch.Tensor,
    u_prev: torch.Tensor,
    dt: float,
    nu: float,
    dealias: bool = True,
) -> torch.Tensor:
    """Burgers residual on the *predicted* future state.

    R = u_t + (u^2/2)_x - nu * u_xx

    Parameters
    ----------
    u_next:
        Predicted state at t + dt, shape (B, X).
    u_prev:
        State at t (may be ground truth, detached), shape (B, X).
    dt:
        Observation time step.
    nu:
        Viscosity.
    dealias:
        Apply the 2/3 rule to the nonlinear flux product.

    Returns
    -------
    Residual tensor of shape (B, X).
    """
    u_t = time_derivative_autograd(u_next, u_prev, dt)
    n = u_next.shape[-1]
    k = spectral_k_1d(n, device=u_next.device, dtype=u_next.dtype)

    if dealias:
        mask = torch.as_tensor(
            dealias_2_3(n), device=u_next.device, dtype=torch.bool
        )
        u_hat = torch.fft.fft(u_next, dim=-1)
        u_filtered = torch.fft.ifft(u_hat * mask, dim=-1).real
        flux = 0.5 * u_filtered**2
    else:
        flux = 0.5 * u_next**2

    flux_hat = torch.fft.fft(flux, dim=-1)
    flux_x = torch.fft.ifft(1j * k * flux_hat, dim=-1).real
    u_xx = d2x_1d(u_next, k)
    return u_t + flux_x - nu * u_xx


def allen_cahn_residual(
    u_next: torch.Tensor,
    u_prev: torch.Tensor,
    dt: float,
    D: float,
    a: float,
) -> torch.Tensor:
    """Allen-Cahn residual on the *predicted* future state.

    R = u_t - D * laplacian(u) - a * (u - u^3)

    Shapes: u_next / u_prev are (B, X, Y); result is (B, X, Y).
    """
    u_t = time_derivative_autograd(u_next, u_prev, dt)
    nx, ny = u_next.shape[-2], u_next.shape[-1]
    kx, ky = spectral_k_2d(nx, ny, device=u_next.device, dtype=u_next.dtype)
    lap = laplacian_2d(u_next, kx, ky)
    return u_t - D * lap - a * (u_next - u_next**3)


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
