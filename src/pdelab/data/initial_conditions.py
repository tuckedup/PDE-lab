"""Deterministic initial-condition generation.

All initial conditions are *band-limited* random fields built from a
truncated Fourier series, so the spectrum is known, recorded, and
reproducible from a seed. The recorded amplitude and spectrum are
written into the dataset manifest so that a reader can verify the
generation settings without re-running the sampler.
"""

from __future__ import annotations

import numpy as np


def fourier_series_1d(
    n: int,
    rng: np.random.Generator,
    n_modes: int = 8,
    amplitude: float = 1.0,
    L: float = 1.0,
    mean: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Band-limited random 1D field.

    Returns the field on ``n`` points of a periodic domain of length L
    (endpoint not duplicated) and a dictionary describing the spectrum.
    """
    x = np.linspace(0.0, L, n, endpoint=False)
    u = np.full(n, float(mean), dtype=np.float64)
    k_used = []
    coeffs = []
    for k in range(1, n_modes + 1):
        a = rng.standard_normal()
        b = rng.standard_normal()
        # 1/sqrt(2) scaling gives a roughly unit-RMS field.
        amp = amplitude / np.sqrt(2.0 * n_modes)
        u += amp * (a * np.cos(2.0 * np.pi * k * x / L) + b * np.sin(2.0 * np.pi * k * x / L))
        k_used.append(k)
        coeffs.append([float(a), float(b)])
    spec = {
        "type": "fourier_series",
        "n_modes": int(n_modes),
        "amplitude": float(amplitude),
        "mean": float(mean),
        "modes": k_used,
        "coefficients": coeffs,
        "realised_rms": float(np.sqrt(np.mean((u - mean) ** 2))),
        "realised_max_abs": float(np.max(np.abs(u))),
    }
    return u, spec


def fourier_series_2d(
    nx: int,
    ny: int,
    rng: np.random.Generator,
    n_modes: int = 4,
    amplitude: float = 0.8,
    L: float = 1.0,
    mean: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Band-limited random 2D field on a periodic square."""
    x = np.linspace(0.0, L, nx, endpoint=False)
    y = np.linspace(0.0, L, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = np.full((nx, ny), float(mean), dtype=np.float64)
    coeffs = []
    for kx in range(n_modes + 1):
        for ky in range(n_modes + 1):
            if kx == 0 and ky == 0:
                continue
            a = rng.standard_normal()
            b = rng.standard_normal()
            amp = amplitude / np.sqrt(2.0 * (n_modes + 1) ** 2)
            u += amp * (
                a * np.cos(2.0 * np.pi * (kx * X + ky * Y) / L)
                + b * np.sin(2.0 * np.pi * (kx * X + ky * Y) / L)
            )
            coeffs.append([int(kx), int(ky), float(a), float(b)])
    spec = {
        "type": "fourier_series_2d",
        "n_modes": int(n_modes),
        "amplitude": float(amplitude),
        "mean": float(mean),
        "coefficients": coeffs,
        "realised_rms": float(np.sqrt(np.mean((u - mean) ** 2))),
        "realised_max_abs": float(np.max(np.abs(u))),
    }
    return u, spec


def gaussian_field_2d(
    nx: int,
    ny: int,
    rng: np.random.Generator,
    correlation_length: float = 0.15,
    amplitude: float = 0.8,
    L: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Smooth random field built from filtered Gaussian noise."""
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=L / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=L / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k2 = KX**2 + KY**2
    noise = rng.standard_normal((nx, ny))
    spec_hat = np.exp(-0.5 * k2 * correlation_length**2) * np.fft.fft2(noise)
    field = np.fft.ifft2(spec_hat).real
    field -= field.mean()
    scale = np.sqrt(np.mean(field**2))
    if scale > 0:
        field = field / scale * amplitude
    spec = {
        "type": "gaussian_filtered",
        "correlation_length": float(correlation_length),
        "amplitude": float(amplitude),
        "realised_rms": float(np.sqrt(np.mean(field**2))),
        "realised_max_abs": float(np.max(np.abs(field))),
    }
    return field, spec
