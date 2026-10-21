import re

for tag, path in [
    ("BEFORE", "/home/atulp/pdelab_audit/validate_before.txt"),
    ("AFTER", "/home/atulp/pdelab_audit/validate_after.txt"),
]:
    text = open(path).read()
    for name in ("burgers.spatial_refinement_smooth",
                 "burgers.spatial_refinement_shock_regime"):
        m = re.search(rf"PASS\s+{re.escape(name)}\n\s+(\{{.*)", text)
        line = m.group(1)
        orders = re.search(r'"(?:observed_orders|pairwise_orders)":\s*\[([^\]]*)\]', line)
        vals = [float(v) for v in orders.group(1).split(",")]
        thr = 1.4 if "smooth" in name else 1.2
        print(f"{tag:6s} {name:42s} pairwise={[round(v, 4) for v in vals]} "
              f"min={min(vals):.4f} mean(fitted)={sum(vals)/len(vals):.4f} "
              f"threshold={thr} -> min:{'PASS' if min(vals) > thr else 'FAIL'} "
              f"fitted:{'PASS' if sum(vals)/len(vals) > thr else 'FAIL'}")
