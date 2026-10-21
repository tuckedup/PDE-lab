"""Shared observation-time scheduling for the PDELab reference solvers.

Why this module exists
----------------------
Every reference solver exposes the same dataset-facing entry point::

    frames = solver.solve_to(u0, t_final, save_times)

``save_times`` is the *observation protocol* of the dataset: the times at
which the reference solution is recorded.  The scheduler therefore has to
guarantee three things, and it must guarantee them for *all* solvers with
one implementation so that the dataset timing protocol cannot drift
between systems:

1. **Exactly one frame per requested observation.**  ``solve_to`` returns
   ``len(save_times)`` frames and frame ``i`` is the state at
   ``save_times[i]``.
2. **The frame is the state *at* the requested time.**  The integrator
   steps exactly onto every requested observation time; it never returns a
   state that belongs to a different time.
3. **No duplicated or missing states.**  Two observation times that fall
   inside a single internal step are still visited separately (the
   internal step is shortened), and the terminal observation is never
   dropped.

The scheduling policy
---------------------
Let ``t_k`` be the current time and ``t_{k+1}`` the next requested
observation.  The gap is ``g = t_{k+1} - t_k`` and the solver's stability
policy provides an upper bound ``dt_max(g)`` on the internal step for the
*current* state.  The gap is covered by an integer number of **equal**
sub-steps::

    n     = ceil(g / dt_max)          (at least one)
    dt    = g / n                    (so dt <= dt_max and n*dt == g)

so the integration lands exactly on ``t_{k+1}``.  This is the
``min(stable_internal_dt, next_observation_gap)`` rule with the
sub-division that makes the landing exact; a short final step is never
needed and the shortened step is never smaller than the solver's own
stability bound divided by the (small) integer ``n``, so it stays stable.

Validation of the request
--------------------------
``save_times`` must be one-dimensional, finite, non-negative, strictly
increasing (hence free of duplicates) and must start at ``0.0``.  The
first entry is treated *deliberately*: the solver starts from the initial
condition at ``t = 0``, so the first returned frame is that state and the
request must ask for ``t = 0`` explicitly.  A request that starts
anywhere else is rejected instead of being silently reinterpreted.
"""

from __future__ import annotations

import numpy as np

#: Absolute tolerance used when comparing observation times.
TIME_TOL = 1e-12

#: Identifier of the scheduling policy, recorded in the dataset manifest.
OBSERVATION_SCHEDULER_ID = "pdelab.observation_schedule.v1"

#: Human readable description of the policy, recorded in the manifest.
OBSERVATION_SCHEDULER_POLICY = (
    "solve_to returns exactly len(save_times) frames; frame i is the state at "
    "save_times[i]; the gap between two requested observations is covered by "
    "ceil(gap/dt_max) equal sub-steps of size gap/ceil(gap/dt_max) so that the "
    "integrator lands exactly on the requested time; dt_max is the solver's own "
    "stability bound evaluated on the current state."
)


def validate_save_times(save_times, t_final: float | None = None) -> np.ndarray:
    """Validate an observation-time request.

    Parameters
    ----------
    save_times:
        The requested observation times.  Must be one-dimensional, finite,
        non-negative, strictly increasing and start at ``0.0``.
    t_final:
        When given, the last observation must not be later than
        ``t_final`` (the solver integrates from ``t = 0`` to ``t_final``).

    Returns
    -------
    numpy.ndarray
        The validated times as a float64 array.

    Raises
    ------
    ValueError
        If the request violates any of the documented conditions.
    """
    try:
        arr = np.asarray(save_times, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"save_times must be a numeric array: {exc}") from exc
    if arr.ndim != 1:
        raise ValueError(
            f"save_times must be one-dimensional, got an array of shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError("save_times must contain at least one observation time")
    if not np.all(np.isfinite(arr)):
        raise ValueError("save_times must be finite")
    if np.any(arr < 0.0):
        raise ValueError("save_times must be non-negative")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(
            "save_times must be strictly increasing (duplicates are not allowed)"
        )
    if abs(float(arr[0])) > TIME_TOL:
        raise ValueError(
            "the first observation time must be 0.0: the solver starts from the "
            "initial condition at t = 0 and the first returned frame is that state"
        )
    if t_final is not None and float(arr[-1]) > float(t_final) + TIME_TOL:
        raise ValueError(
            f"the last observation time {float(arr[-1])!r} is later than "
            f"t_final = {float(t_final)!r}"
        )
    return arr


def substep_plan(gap: float, dt_max: float) -> tuple[int, float]:
    """Plan the sub-steps that cover ``gap`` and land exactly on its end.

    Parameters
    ----------
    gap:
        The distance to the next observation time.  Must be positive.
    dt_max:
        The solver's stability bound for the current state.  Must be
        positive and finite.

    Returns
    -------
    tuple[int, float]
        ``(n_substeps, dt)`` with ``n_substeps >= 1``, ``dt <= dt_max`` and
        ``n_substeps * dt == gap`` up to floating-point round-off.
    """
    if not np.isfinite(gap) or gap <= 0.0:
        raise ValueError(f"gap must be a positive finite number, got {gap!r}")
    if not np.isfinite(dt_max) or dt_max <= 0.0:
        raise ValueError(f"dt_max must be a positive finite number, got {dt_max!r}")
    # The 1e-12 slack absorbs the round-off of gap/dt_max so that a gap
    # which is an exact multiple of dt_max does not produce one extra,
    # needlessly short step.
    n_sub = int(np.ceil(gap / dt_max - 1e-12))
    n_sub = max(n_sub, 1)
    return n_sub, gap / n_sub


def scheduler_manifest_entry() -> dict[str, str]:
    """The scheduling policy, as recorded in the dataset manifest."""
    return {
        "observation_scheduler": OBSERVATION_SCHEDULER_ID,
        "observation_scheduler_policy": OBSERVATION_SCHEDULER_POLICY,
        "observation_time_tolerance": f"{TIME_TOL:g}",
    }
