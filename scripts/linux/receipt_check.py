"""Receipt check: confirm the observation-time scheduling defect (audit finding A).

This is an *independent* check of the finding; no solver source is modified.

Method
------
``solve_to(u0, t_final, save_times)`` is supposed to return one state per
requested observation time, each state taken at exactly that time.  To make
the time of every returned frame directly observable we replace the solver's
integrator with a *clock-valued* step ``u -> u + dt`` (so the field itself
carries the elapsed time) and force the internal step size to a known
``dt = 0.005``.

With the defect present the scheduler tests ``save_times[len(out) - 1]``
against the current time, so the frame produced by the *first* internal step
is stored in the slot of the first *positive* observation and every later
observation lags by one slot; the terminal frame is dropped.
"""

from __future__ import annotations

import numpy as np

from pdelab.solvers.allen_cahn import AllenCahn2DSpectral
from pdelab.solvers.burgers_weno import Burgers1DWeno5

SAVE_TIMES = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
T_FINAL = float(SAVE_TIMES[-1])
DT = 0.005


def clock_of(frame: np.ndarray) -> float:
    """Elapsed time encoded in a clock-valued frame."""
    return float(np.mean(frame))


def uniformity(frame: np.ndarray) -> float:
    """Spread of the frame values (0.0 for a perfectly uniform clock)."""
    return float(np.max(frame) - np.min(frame))


def run_case(name: str, solver, u0: np.ndarray) -> bool:
    # Clock-valued integrator: the state *is* the clock.
    solver.step = lambda u, dt: u + dt  # type: ignore[method-assign]
    # Known internal step size.
    solver.stable_dt = lambda *args, **kwargs: DT  # type: ignore[method-assign]

    frames = solver.solve_to(u0, T_FINAL, SAVE_TIMES)
    clocks = np.array([clock_of(f) for f in frames])
    spreads = np.array([uniformity(f) for f in frames])

    print(f"=== {name} ===")
    print(f"  requested times      : {[float(t) for t in SAVE_TIMES]}")
    print(f"  returned frame count : {len(frames)} (expected {len(SAVE_TIMES)})")
    print(f"  returned clock values: {[round(float(c), 12) for c in clocks]}")
    print(f"  frame uniformity     : max spread = {max(spreads):.3e} "
          f"(0 means each frame is a clean clock value)")
    print(f"  solver internal steps: {solver._last_steps}, dt = {solver._last_dt}")

    if len(frames) != len(SAVE_TIMES):
        print("  MISMATCH: wrong number of frames returned")
        print()
        return False

    err = np.abs(clocks - SAVE_TIMES)
    ok = bool(np.all(err <= 1e-12))
    for req, got, e in zip(SAVE_TIMES, clocks, err):
        flag = "ok " if e <= 1e-12 else "OFF"
        print(f"    [{flag}] requested {req:.6f} -> returned {got:.6f}  (err {e:.6f})")
    print(f"  frames match requested times: {ok}")
    print()
    return ok


def main() -> int:
    print(f"save_times = {[float(t) for t in SAVE_TIMES]}, t_final = {T_FINAL}, forced dt = {DT}")
    print()

    burgers = Burgers1DWeno5(16)
    ok_b = run_case("Burgers1DWeno5(16)", burgers, np.zeros(16, dtype=np.float64))

    allen = AllenCahn2DSpectral(8, 8)
    ok_a = run_case("AllenCahn2DSpectral(8, 8)", allen, np.zeros((8, 8), dtype=np.float64))

    print("SUMMARY")
    print(f"  Burgers1DWeno5      matches requested times: {ok_b}")
    print(f"  AllenCahn2DSpectral matches requested times: {ok_a}")
    if ok_b and ok_a:
        print("  RESULT: no defect observed (scheduler honours observation times)")
        return 0
    print("  RESULT: DEFECT CONFIRMED (returned frames do not match requested times)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
