"""Seeding helpers.

Reproducibility contract
------------------------
A run is identified by its *training seed*. For that to be true, every
source of randomness that affects the run must be seeded **before** the
model is constructed:

* the parameter initialisation,
* the data shuffling order,
* any stochastic layer or dropout.

The seeding therefore happens in two places, and both must use the same
seed:

1. ``seed_everything`` is called by the experiment runner *before* the
   model is built, so the initialisation is a deterministic function of
   the seed.
2. ``Trainer.train`` re-seeds with the same value, so the data order
   and the stochastic layers are deterministic too.

Seeding only inside the trainer (as an earlier version did) left the
initialisation at the mercy of the process-level RNG, so two runs with
the same seed produced different networks.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed every RNG that can influence a training run."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_fingerprint(model: torch.nn.Module) -> str:
    """A short hash of the model initialisation.

    Recorded in every run manifest so that a specific trained network
    can be identified exactly, independently of the seed label.
    Complex tensors (the FNO's spectral weights) are hashed as complex
    values, not cast to real, so the imaginary part is not discarded.
    """
    import hashlib

    h = hashlib.sha256()
    for name, param in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        t = param.detach().cpu().contiguous()
        if t.is_complex():
            h.update(t.view(torch.float32).numpy().tobytes())
        else:
            h.update(t.to(torch.float32).numpy().tobytes())
    return h.hexdigest()[:16]
