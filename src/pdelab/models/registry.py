"""Model construction from configuration.

All model conditions share the same one-step interface, the same input
channel layout and the same PDE-parameter conditioning, so that the only
differences between conditions are the architecture and the presence of
the physics penalty.
"""

from __future__ import annotations

from typing import Any

from .base import TimeAdvanceModel
from .fno import FNO1d, FNO2d
from .unet import PeriodicUNet1d, PeriodicUNet2d


def build_model(
    architecture: str,
    dimension: int,
    history: int = 1,
    n_params: int = 0,
    **kwargs: Any,
) -> TimeAdvanceModel:
    """Instantiate a model by name.

    Parameters
    ----------
    architecture:
        One of ``fno1d``, ``fno2d``, ``unet1d``, ``unet2d``.
    dimension:
        Spatial dimension of the problem (1 or 2).
    """
    arch = architecture.lower()
    if arch == "fno1d":
        return FNO1d(history=history, n_params=n_params, **kwargs)
    if arch == "fno2d":
        return FNO2d(history=history, n_params=n_params, **kwargs)
    if arch == "unet1d":
        return PeriodicUNet1d(history=history, n_params=n_params, **kwargs)
    if arch == "unet2d":
        return PeriodicUNet2d(history=history, n_params=n_params, **kwargs)
    raise ValueError(f"unknown architecture {architecture!r}")


def architecture_defaults(architecture: str) -> dict[str, Any]:
    """Conservative pilot defaults for a given architecture."""
    arch = architecture.lower()
    if arch == "fno1d":
        return {"modes": 16, "width": 16, "n_layers": 4}
    if arch == "fno2d":
        return {"modes_x": 8, "modes_y": 8, "width": 16, "n_layers": 4}
    if arch == "unet1d":
        return {"width": 16, "n_levels": 3}
    if arch == "unet2d":
        return {"width": 16, "n_levels": 3}
    raise ValueError(f"unknown architecture {architecture!r}")


def build_model_from_config(config: dict[str, Any], dimension: int) -> TimeAdvanceModel:
    """Build a model from a resolved experiment configuration block."""
    arch = config["architecture"]
    history = int(config.get("history", 1))
    n_params = int(config.get("n_params", 0))
    kwargs = {k: v for k, v in config.items() if k not in
              ("architecture", "history", "n_params")}
    return build_model(
        arch, dimension=dimension, history=history, n_params=n_params, **kwargs
    )
