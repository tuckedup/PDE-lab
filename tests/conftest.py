"""Shared pytest fixtures for the PDELab test suite."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pdelab.config import project_root


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20261019)


@pytest.fixture(autouse=True)
def deterministic():
    """Make every test reproducible."""
    torch.manual_seed(1234)
    np.random.seed(1234)
    yield


@pytest.fixture(scope="session")
def root():
    return project_root()


@pytest.fixture(scope="session")
def burgers_ic():
    """A smooth periodic 1D initial condition."""
    x = np.linspace(0.0, 1.0, 64, endpoint=False)
    return 0.8 * np.cos(2 * np.pi * x) + 0.3 * np.cos(4 * np.pi * x + 0.5)
