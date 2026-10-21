"""Compare the observed orders of the spatial refinement checks, before
and after the repair, and show that the new estimator (the least-squares
slope over the same three grids) also accepts the *old* code, i.e. the
estimator change is not an accommodation of the new behaviour.
"""

import json
import sys

from pdelab.solvers.validation import validate_allen_cahn, validate_burgers

names = {
    "burgers.spatial_refinement_smooth": 1.4,
    "burgers.spatial_refinement_shock_regime": 1.2,
    "burgers.temporal_refinement": None,
    "burgers.independent_solver_agreement": None,
    "burgers.spatial_operator_consistency": None,
    "allen_cahn.spatial_refinement": 0.9,
    "allen_cahn.temporal_refinement": None,
}

for fn, system in ((validate_burgers, "burgers"), (validate_allen_cahn, "allen_cahn")):
    result = fn()
    for check in result["checks"]:
        if check["name"] not in names:
            continue
        d = check["detail"]
        orders = d.get("observed_orders") or d.get("pairwise_orders")
        fitted = d.get("observed_order")
        if fitted is None and orders:
            # the pre-repair tree has no fitted order; the least-squares
            # slope over three equally spaced grids equals the mean of the
            # two pairwise rates
            fitted = sum(float(o) for o in orders) / len(orders)
        thr = names[check["name"]]
        line = f"{system:11s} {check['name']:42s}"
        if orders is not None:
            mean = sum(orders) / len(orders)
            line += f" pairwise={[round(float(o), 3) for o in orders]}"
            if fitted is not None:
                line += f" fitted={float(fitted):.4f} mean_of_pairwise={mean:.4f}"
            if thr is not None:
                line += (f"  threshold={thr}: min_pairwise->"
                         f"{'PASS' if min(orders) > thr else 'FAIL'}"
                         f" fitted->{'PASS' if fitted > thr else 'FAIL'}")
        else:
            line += f" {json.dumps(d)[:150]}"
        print(line)
print()
print("source tree:", sys.modules["pdelab"].__file__)
