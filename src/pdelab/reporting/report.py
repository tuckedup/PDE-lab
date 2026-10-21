"""Build result tables from stored artifacts.

Every number in the report is read from a stored artifact (a run
manifest, an evaluation file, a benchmark file). Nothing is typed in by
hand, so the report cannot drift from what was actually measured.

Cohorts
-------
A target is evaluated against a **declared cohort**: a frozen set of
runs (system, architecture, condition, seeds) together with the dataset
hash, the evaluator identity, the protocol and the resolved settings
that every one of them must match.  The cohort is declared in
``campaigns/evidence-cohort/cohort.yaml``.

* A run that does not match the cohort identity is **excluded with a
  reason**; it is never silently substituted.
* An incomplete cohort yields ``not_run`` or ``incomplete_cohort`` -
  never ``met``.
* The declared reference model for the inference targets is used as
  declared.  If it is missing the target fails explicitly; the best
  available benchmark is never substituted for it.

Evidence states
---------------
Every artifact carries an evidence state with a reason:

``valid``
    The artifact is intact, newer than the checkpoint it evaluates, and
    its content identity (checkpoint, dataset, normalisation, evaluator
    implementation, resolved settings, protocol) matches both the
    cohort and the code on disk.
``stale``
    The artifact is older than the checkpoint it claims to evaluate, or
    the checkpoint / dataset it was computed from has been replaced.
``invalid_protocol``
    The artifact was produced under a different evaluator, protocol or
    resolved setting, or it predates the identity-bound protocol.
``failed``
    The artifact is corrupt, or the run it belongs to failed.
``interrupted``
    The run it belongs to did not finish.
``not_run``
    There is no artifact.
``not_comparable``
    The artifact exists but the metric cannot be compared (for example
    a zero residual denominator).

A target whose evidence is not ``valid`` reports that state and *no
number*.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..config import load_yaml, project_root
from ..evaluation.metrics import (
    evaluator_identity,
    read_digest_sidecar,
    relative_residual_improvement,
    sha256_file,
    write_json_atomic,
    write_text_atomic,
)

#: The evidence states an artifact can be in.
EVIDENCE_STATES = (
    "valid",
    "stale",
    "invalid_protocol",
    "failed",
    "interrupted",
    "not_run",
    "not_comparable",
)

#: The states a *target* can be in.
TARGET_STATUSES = ("met", "missed", "incomplete_cohort", *EVIDENCE_STATES)

#: When several members of a cohort disagree, the most severe state
#: wins.  This order defines "most severe"; ``valid`` is the least
#: severe state.
_STATE_PRECEDENCE = (
    "failed",
    "invalid_protocol",
    "stale",
    "interrupted",
    "not_comparable",
    "incomplete_cohort",
    "not_run",
    "valid",
)

#: Where the cohort declaration lives.
DEFAULT_COHORT_PATH = "campaigns/evidence-cohort/cohort.yaml"

#: The residual metric that the primary (first-order) evaluator feeds.
PRIMARY_RESIDUAL_KEY = "residual_rms_implicit_euler_first_order"
CENTRED_RESIDUAL_KEY = "residual_rms_centred_first_derivative"


def _worst_state(states: list[str]) -> str:
    """The most severe of the given evidence states."""
    known = [s for s in states if s in _STATE_PRECEDENCE and s != "valid"]
    if not known:
        return "not_run"
    return min(known, key=lambda s: _STATE_PRECEDENCE.index(s))


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read a JSON artifact.

    Returns ``(payload, error)``.  A missing file is *not* an error
    (``payload`` is ``None``); a corrupt file is reported, never
    dropped silently.
    """
    if not path.exists():
        return None, None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh), None
    except json.JSONDecodeError as exc:
        return None, f"not valid JSON ({exc})"
    except OSError as exc:
        return None, f"could not be read ({exc})"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value in (float("inf"), float("-inf")):
            return str(value)
        return f"{value:.{digits}g}"
    return str(value)


def collect_runs(runs_dir: str | Path) -> list[dict[str, Any]]:
    """Collect every run directory and its artifacts.

    A run whose artifacts are missing or corrupt is still collected:
    the error is recorded so that the report can show *why* a run
    contributes nothing.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = project_root() / runs_dir
    out: list[dict[str, Any]] = []
    if not runs_dir.exists():
        return out
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        manifest, manifest_error = _read_json(d / "manifest.json")
        evaluation, evaluation_error = _read_json(d / "evaluation.json")
        benchmark, benchmark_error = _read_json(d / "benchmark_matched.json")
        out.append({
            "dir": d,
            "run_id": d.name,
            "manifest": manifest,
            "manifest_error": manifest_error,
            "evaluation": evaluation,
            "evaluation_error": evaluation_error,
            "benchmark": benchmark,
            "benchmark_error": benchmark_error,
        })
    return out


def _run_identity(run: dict[str, Any]) -> dict[str, Any]:
    """The declared identity of a run, from its manifest."""
    manifest = run.get("manifest") or {}
    cfg = manifest.get("experiment") or {}
    return {
        "system": cfg.get("system"),
        "architecture": cfg.get("architecture"),
        "condition": cfg.get("condition"),
        "lambda_pde": cfg.get("lambda_pde"),
        "seed": cfg.get("seed"),
        "run_id": manifest.get("run_id") or run.get("run_id"),
    }


def assess_run_evidence(
    run: dict[str, Any],
    cohort: dict[str, Any] | None = None,
    evaluator_now: dict[str, Any] | None = None,
    artifact: str = "evaluation",
) -> dict[str, Any]:
    """The evidence state of one artifact of one run, with a reason."""
    evaluator_now = evaluator_now or evaluator_identity()
    reasons: list[str] = []
    state = "valid"

    def fail(new_state: str, reason: str) -> None:
        nonlocal state
        reasons.append(reason)
        if _STATE_PRECEDENCE.index(new_state) < _STATE_PRECEDENCE.index(state):
            state = new_state

    manifest = run.get("manifest")
    if run.get("manifest_error"):
        fail("failed", f"manifest.json is corrupt: {run['manifest_error']}")
        return {"state": state, "reason": reasons[0], "reasons": reasons}
    if manifest is None:
        fail("not_run", "no manifest.json")
        return {"state": state, "reason": reasons[0], "reasons": reasons}

    status = manifest.get("status")
    if status != "completed":
        if status == "failed":
            fail("failed", f"the run failed: {manifest.get('error', 'no reason recorded')}")
        elif status in ("running", "interrupted"):
            fail("interrupted", f"the run did not finish (status {status!r})")
        elif status == "blocked":
            fail("failed", f"the run was blocked before it started: {manifest.get('block_reason')}")
        else:
            fail("not_run", f"the run did not complete (status {status!r})")

    artifact_name = "evaluation.json" if artifact == "evaluation" else "benchmark_matched.json"
    path = run["dir"] / artifact_name
    payload = run.get(artifact)
    if run.get(f"{artifact}_error"):
        fail("failed", f"{artifact_name} is corrupt: {run[f'{artifact}_error']}")
        return {"state": state, "reason": reasons[0], "reasons": reasons}
    if payload is None:
        fail("not_run", f"the run {run.get('run_id')} has no {artifact_name}")
        return {"state": state, "reason": reasons[0], "reasons": reasons}

    # ---- integrity of the artifact bytes ----
    sidecar = read_digest_sidecar(path)
    if sidecar is not None:
        actual = sha256_file(path)
        if sidecar != actual:
            fail(
                "failed",
                f"{artifact_name} does not match its recorded digest "
                "(the file was modified or truncated)",
            )

    # ---- content identity ----
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        fail(
            "invalid_protocol",
            f"{artifact_name} records no content identity; it was produced "
            "before the identity-bound protocol and cannot be compared",
        )
        # The identity-dependent checks are skipped, but the freshness
        # checks below still run: an artifact that is older than its
        # checkpoint is stale whether or not it carries an identity.
        identity = {}

    recorded_evaluator = (identity.get("evaluator") or {}).get("sha256")
    if recorded_evaluator != evaluator_now["sha256"]:
        fail(
            "invalid_protocol",
            "the evaluator implementation has changed since the artifact was "
            f"produced (recorded {recorded_evaluator}, current "
            f"{evaluator_now['sha256']})",
        )
    if not identity.get("resolved_settings_sha256"):
        fail("invalid_protocol", f"{artifact_name} records no resolved settings")
    if not (identity.get("protocol") or {}).get("id"):
        fail("invalid_protocol", f"{artifact_name} records no protocol identity")

    # ---- checkpoint and dataset freshness ----
    ckpt_path = run["dir"] / "best.pt"
    recorded_ckpt = identity.get("checkpoint_sha256")
    if ckpt_path.exists():
        actual_ckpt = sha256_file(ckpt_path)
        if recorded_ckpt and recorded_ckpt != actual_ckpt:
            fail(
                "stale",
                "the checkpoint has been replaced since the artifact was produced "
                f"(recorded {recorded_ckpt[:12]}, on disk {actual_ckpt[:12]})",
            )
        if path.exists() and path.stat().st_mtime < ckpt_path.stat().st_mtime:
            fail(
                "stale",
                f"{artifact_name} is older than the checkpoint it evaluates",
            )
    else:
        fail("not_run", "no checkpoint (best.pt) to bind the artifact to")

    dataset = identity.get("dataset") or {}
    dataset_path = dataset.get("path")
    recorded_ds = dataset.get("sha256")
    if dataset_path and Path(dataset_path).exists():
        actual_ds = sha256_file(dataset_path)
        if recorded_ds and recorded_ds != actual_ds:
            fail(
                "stale",
                "the evaluated dataset file has changed since the artifact was "
                "produced",
            )

    # ---- the cohort identity ----
    if cohort:
        declared_evaluator = (cohort.get("evaluator") or {}).get("sha256")
        if declared_evaluator and recorded_evaluator != declared_evaluator:
            fail(
                "invalid_protocol",
                "the artifact's evaluator does not match the cohort's declared "
                "evaluator",
            )
        declared_evaluator_id = (cohort.get("evaluator") or {}).get("id")
        recorded_evaluator_id = (identity.get("evaluator") or {}).get("id")
        if declared_evaluator_id and recorded_evaluator_id != declared_evaluator_id:
            fail(
                "invalid_protocol",
                "the artifact's evaluator id does not match the cohort's "
                "declared evaluator id",
            )
        declared_protocol = (cohort.get("protocol") or {}).get("id")
        recorded_protocol = (identity.get("protocol") or {}).get("id")
        if declared_protocol and recorded_protocol != declared_protocol:
            fail(
                "invalid_protocol",
                "the artifact was produced under a different protocol "
                f"({recorded_protocol!r} instead of {declared_protocol!r})",
            )
        system = payload.get("system") or (_run_identity(run)["system"])
        declared_dataset = ((cohort.get("datasets") or {}).get(system) or {}).get(
            f"{(dataset.get('split') or payload.get('split') or 'test')}_sha256"
        )
        if declared_dataset and recorded_ds and recorded_ds != declared_dataset:
            fail(
                "invalid_protocol",
                "the artifact was computed on a dataset that the cohort does "
                "not declare",
            )

    return {
        "state": state,
        "reason": reasons[0] if reasons else "the artifact is intact and matches its identity",
        "reasons": reasons,
        "identity": identity,
    }


def _member_runs(
    member: dict[str, Any],
    runs: list[dict[str, Any]],
    cohort: dict[str, Any],
    evaluator_now: dict[str, Any],
) -> list[dict[str, Any]]:
    """The runs of one cohort member, with their evidence state.

    A run that does not exist, or whose manifest does not match the
    member's declared identity, is excluded with an explicit reason.
    """
    out: list[dict[str, Any]] = []
    run_ids = member.get("run_ids")
    if not run_ids:
        out.append({
            "run_id": None,
            "state": "not_run",
            "reason": f"the cohort member {member.get('id')!r} declares no run ids",
            "reasons": [],
            "seed": None,
        })
        return out
    for run_id in run_ids:
        run = next((r for r in runs if r["run_id"] == run_id), None)
        if run is None:
            out.append({
                "run_id": run_id,
                "state": "not_run",
                "reason": f"the run {run_id} declared by the cohort does not exist",
                "reasons": [],
                "seed": None,
            })
            continue
        identity = _run_identity(run)
        mismatch = [
            key
            for key in ("system", "architecture", "condition", "lambda_pde", "seed")
            if member.get(key) is not None and member.get(key) != identity.get(key)
        ]
        if mismatch:
            out.append({
                "run_id": run_id,
                "state": "invalid_protocol",
                "reason": (
                    f"the run {run_id} does not match the cohort member's "
                    f"declared identity ({', '.join(mismatch)})"
                ),
                "reasons": [],
                "seed": identity.get("seed"),
            })
            continue
        evidence = assess_run_evidence(run, cohort, evaluator_now)
        out.append({
            "run_id": run_id,
            "state": evidence["state"],
            "reason": evidence["reason"],
            "reasons": evidence["reasons"],
            "seed": identity.get("seed"),
            "run": run,
        })
    return out


def _residual_value(evaluation: dict[str, Any], key: str) -> float | None:
    value = evaluation.get(key)
    if value is None and key == PRIMARY_RESIDUAL_KEY:
        value = evaluation.get("residual_error_physical_rms")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _satisfies(value: float, threshold: Any, comparison: str) -> bool:
    if threshold is None:
        return False
    if comparison == "<":
        return value < threshold
    if comparison == "<=":
        return value <= threshold
    if comparison == ">":
        return value > threshold
    if comparison == ">=":
        return value >= threshold
    if comparison == "==":
        return value == threshold
    return False


def build_target_table(
    runs: list[dict[str, Any]],
    targets: dict[str, Any],
    cohort: dict[str, Any] | None = None,
    evaluator_now: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate each declared target against the declared cohort."""
    evaluator_now = evaluator_now or evaluator_identity()
    cohort = cohort or {}
    bindings = {b.get("name"): b for b in (cohort.get("targets") or [])}
    members = {m.get("id"): m for m in (cohort.get("members") or [])}

    rows: list[dict[str, Any]] = []
    for target in targets.get("targets", []):
        name = target["name"]
        metric = target["metric"]
        threshold = target.get("threshold")
        comparison = target.get("comparison", "<")
        row: dict[str, Any] = {
            "target": name,
            "metric": metric,
            "threshold": threshold,
            "comparison": comparison,
            "units": target.get("units"),
            "status": "not_run",
            "measured": None,
            "run_id": None,
            "checkpoint": None,
            "test_dataset_sha256": None,
            "protocol": None,
            "note": "",
            "cohort": cohort.get("cohort_id"),
            "evidence": [],
            "values": [],
        }

        if not cohort:
            row["note"] = (
                "no cohort declaration was found; a target cannot be met "
                "without one"
            )
            rows.append(row)
            continue
        binding = bindings.get(name)
        if binding is None:
            row["note"] = (
                "the target is not bound to a declared cohort, so it cannot "
                "be evaluated"
            )
            rows.append(row)
            continue

        kind = binding.get("metric", metric)
        if kind == "relative_residual_improvement":
            _fill_improvement(row, binding, members, runs, cohort, evaluator_now, threshold, comparison)
        elif kind in ("throughput_ratio", "memory_reduction_percent"):
            _fill_inference(row, binding, runs, cohort, evaluator_now, targets, threshold, comparison)
        else:
            _fill_metric(row, binding, members, runs, cohort, evaluator_now, threshold, comparison)
        rows.append(row)
    return rows


def _problem_note(problems: list[dict[str, Any]]) -> str:
    """A note that lists every reason, not only the most severe one."""
    parts = []
    for p in problems:
        reasons = p.get("reasons") or [p.get("reason", "")]
        for reason in reasons:
            label = p.get("run_id") or "cohort"
            parts.append(f"{label}: {reason}")
    return "; ".join(parts)


def _fill_metric(
    row: dict[str, Any],
    binding: dict[str, Any],
    members: dict[str, Any],
    runs: list[dict[str, Any]],
    cohort: dict[str, Any],
    evaluator_now: dict[str, Any],
    threshold: Any,
    comparison: str,
) -> None:
    """A single-metric target (the rollout nRMSE)."""
    values: list[tuple[str, float]] = []
    problems: list[dict[str, Any]] = []
    used: list[str] = []
    for member_id in binding.get("members", []):
        member = members.get(member_id)
        if member is None:
            problems.append({
                "run_id": None,
                "state": "invalid_protocol",
                "reason": f"the cohort declares the unknown member {member_id!r}",
                "reasons": [],
            })
            continue
        for entry in _member_runs(member, runs, cohort, evaluator_now):
            if entry["state"] != "valid":
                problems.append(entry)
                continue
            evaluation = entry["run"]["evaluation"]
            value = ((evaluation.get("nrmse") or {}).get("mean"))
            if value is None:
                problems.append({
                    "run_id": entry["run_id"],
                    "state": "invalid_protocol",
                    "reason": "the artifact records no nRMSE",
                    "reasons": [],
                })
                continue
            values.append((entry["run_id"], float(value)))
            used.append(entry["run_id"])

    row["evidence"] = [
        {
            "run_id": p["run_id"],
            "state": p["state"],
            "reason": p["reason"],
            "reasons": p.get("reasons") or [],
        }
        for p in problems
    ]
    row["values"] = [{"run_id": r, "value": v} for r, v in values]

    if problems:
        state = _worst_state([p["state"] for p in problems])
        row["status"] = state
        row["note"] = _problem_note(problems)
        return
    if not values:
        row["status"] = "not_run"
        row["note"] = "no run of the declared cohort has been evaluated"
        return

    measured = sum(v for _r, v in values) / len(values)
    verdict_ok = all(_satisfies(v, threshold, comparison) for _r, v in values)
    row["measured"] = measured
    row["status"] = "met" if verdict_ok else "missed"
    row["run_id"] = used[0] if len(used) == 1 else ", ".join(used)
    first = next((r for r in runs if r["run_id"] == used[0]), None)
    if first is not None:
        row["checkpoint"] = ((first["evaluation"] or {}).get("checkpoint"))
        row["test_dataset_sha256"] = ((first["evaluation"] or {}).get("test_dataset_sha256"))
        row["protocol"] = (first["evaluation"] or {}).get("protocol")
    row["note"] = (
        f"cohort mean over {len(values)} run(s): "
        + ", ".join(f"{r}={v:.4g}" for r, v in values)
        + f"; the verdict requires every declared member to satisfy "
        f"'{comparison} {threshold}'"
    )


def _fill_improvement(
    row: dict[str, Any],
    binding: dict[str, Any],
    members: dict[str, Any],
    runs: list[dict[str, Any]],
    cohort: dict[str, Any],
    evaluator_now: dict[str, Any],
    threshold: Any,
    comparison: str,
) -> None:
    """A relative residual improvement target (baseline vs proposed)."""
    baseline_member = members.get(binding.get("baseline"))
    proposed_member = members.get(binding.get("proposed"))
    if baseline_member is None or proposed_member is None:
        row["status"] = "invalid_protocol"
        row["note"] = "the cohort does not declare the baseline/proposed pair"
        return

    base_entries = {
        e["seed"]: e for e in _member_runs(baseline_member, runs, cohort, evaluator_now)
    }
    prop_entries = {
        e["seed"]: e for e in _member_runs(proposed_member, runs, cohort, evaluator_now)
    }
    seeds = sorted(set(base_entries) | set(prop_entries), key=lambda s: (s is None, s))

    values: list[tuple[str, float]] = []
    problems: list[dict[str, Any]] = []
    for seed in seeds:
        base = base_entries.get(seed)
        prop = prop_entries.get(seed)
        for role, entry in (("baseline", base), ("proposed", prop)):
            if entry is None:
                problems.append({
                    "run_id": None,
                    "state": "not_run",
                    "reason": f"the {role} run for seed {seed} is not declared",
                    "reasons": [],
                })
            elif entry["state"] != "valid":
                problems.append(entry)
        if base is None or prop is None:
            continue
        if base["state"] != "valid" or prop["state"] != "valid":
            continue
        b = _residual_value(base["run"]["evaluation"], PRIMARY_RESIDUAL_KEY)
        p = _residual_value(prop["run"]["evaluation"], PRIMARY_RESIDUAL_KEY)
        if b is None or p is None:
            problems.append({
                "run_id": prop["run_id"],
                "state": "invalid_protocol",
                "reason": "the artifact records no primary residual",
                "reasons": [],
            })
            continue
        value = relative_residual_improvement(b, p)
        if value != value:
            row["status"] = "not_comparable"
            row["note"] = "the baseline residual denominator is zero or undefined"
            return
        values.append((f"seed{seed}", float(value)))

    row["evidence"] = [
        {
            "run_id": p["run_id"],
            "state": p["state"],
            "reason": p["reason"],
            "reasons": p.get("reasons") or [],
        }
        for p in problems
    ]
    row["values"] = [{"run_id": r, "value": v} for r, v in values]

    if problems:
        row["status"] = _worst_state([p["state"] for p in problems])
        row["note"] = _problem_note(problems)
        return
    if not values:
        row["status"] = "not_run"
        row["note"] = "no matched baseline/proposed pair of the cohort has been evaluated"
        return

    measured = sum(v for _r, v in values) / len(values)
    verdict_ok = all(_satisfies(v, threshold, comparison) for _r, v in values)
    row["measured"] = measured
    row["status"] = "met" if verdict_ok else "missed"
    row["run_id"] = ", ".join(f"seed{k}" for k in sorted(prop_entries))
    prop_run = next((r for r in runs if r["run_id"] in prop_entries), None)
    if prop_run is not None:
        row["checkpoint"] = ((prop_run["evaluation"] or {}).get("checkpoint"))
        row["test_dataset_sha256"] = (
            (prop_run["evaluation"] or {}).get("test_dataset_sha256")
        )
        row["protocol"] = {
            "baseline": sorted(base_entries),
            "proposed": sorted(prop_entries),
            "residual_evaluator": "implicit_euler_first_order (primary)",
            "residual_support": "interior snapshots only (shared)",
        }
    detail = []
    for seed in seeds:
        base = base_entries.get(seed)
        prop = prop_entries.get(seed)
        if base is None or prop is None:
            continue
        bc = _residual_value(base["run"]["evaluation"], CENTRED_RESIDUAL_KEY)
        pc = _residual_value(prop["run"]["evaluation"], CENTRED_RESIDUAL_KEY)
        if bc is not None and pc is not None:
            detail.append(
                f"seed{seed} centred-evaluator improvement "
                f"{relative_residual_improvement(bc, pc):+.2f}%"
            )
    row["note"] = (
        "primary (first-order, implicit Euler) improvement per seed: "
        + ", ".join(f"{r}={v:+.2f}%" for r, v in values)
        + f"; the verdict requires every seed to satisfy '{comparison} {threshold}'"
        + ("; " + "; ".join(detail) if detail else "")
    )


def _fill_inference(
    row: dict[str, Any],
    binding: dict[str, Any],
    runs: list[dict[str, Any]],
    cohort: dict[str, Any],
    evaluator_now: dict[str, Any],
    targets: dict[str, Any],
    threshold: Any,
    comparison: str,
) -> None:
    """An inference target, evaluated on the *declared* reference model."""
    reference = binding.get("reference")
    if not reference:
        row["status"] = "not_run"
        row["note"] = "the cohort declares no reference run for this target"
        return
    declared_reference = targets.get("inference_reference_model")
    if declared_reference and reference != declared_reference:
        row["status"] = "invalid_protocol"
        row["note"] = (
            "the cohort's reference run does not match the declared reference "
            f"model ({declared_reference!r})"
        )
        return
    run = next((r for r in runs if r["run_id"] == reference), None)
    if run is None:
        row["status"] = "not_run"
        row["note"] = f"the declared reference run {reference} does not exist"
        return
    evidence = assess_run_evidence(
        run, cohort, evaluator_now, artifact="benchmark"
    )
    row["evidence"] = [
        {"run_id": reference, "state": evidence["state"], "reason": evidence["reason"]}
    ]
    if evidence["state"] != "valid":
        row["status"] = evidence["state"]
        row["note"] = "; ".join(evidence.get("reasons") or [evidence["reason"]])
        return
    value = (run["benchmark"] or {}).get(row["metric"])
    if value is None:
        row["status"] = "not_comparable"
        row["note"] = f"the benchmark artifact records no {row['metric']}"
        return
    row["measured"] = float(value)
    row["status"] = "met" if _satisfies(float(value), threshold, comparison) else "missed"
    row["run_id"] = reference
    row["checkpoint"] = (run["benchmark"] or {}).get("checkpoint")
    row["protocol"] = {
        "workload": (run["benchmark"] or {}).get("workload"),
        "isolation": (run["benchmark"] or {}).get("isolation"),
        "variant": (run["benchmark"] or {}).get(
            "throughput_best_variant"
            if row["metric"] == "throughput_ratio"
            else "memory_best_variant"
        ),
        "reference_model_declared": reference,
    }
    row["note"] = (
        f"measured on the declared reference run {reference}; no other run is "
        "substituted for it"
    )


def _artifact_timestamp(runs: list[dict[str, Any]]) -> str:
    """The newest modification time of any artifact the report draws on.

    The report is dated from its *inputs*, not from the wall clock, so
    that regenerating it over unchanged artifacts produces an identical
    file. That keeps the generated table stable in version control and
    makes it obvious when a report actually changed.
    """
    newest = 0.0
    for r in runs:
        for key in ("manifest.json", "evaluation.json", "benchmark_matched.json"):
            p = r["dir"] / key
            if p.exists():
                newest = max(newest, p.stat().st_mtime)
    if newest == 0.0:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(newest))


def build_report(
    output_dir: str | Path = "results",
    make_plots: bool = True,
    runs_dir: str | Path | None = None,
    cohort_path: str | Path | None = None,
    write_docs: bool = True,
) -> dict[str, Any]:
    """Build the full report from artifacts."""
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    targets_path = project_root() / "configs" / "targets.yaml"
    targets = load_yaml(targets_path) if targets_path.exists() else {"targets": []}

    cohort_file = Path(cohort_path) if cohort_path else project_root() / DEFAULT_COHORT_PATH
    if not cohort_file.is_absolute():
        cohort_file = project_root() / cohort_file
    cohort = load_yaml(cohort_file) if cohort_file.exists() else {}

    runs = collect_runs(Path(runs_dir) if runs_dir else project_root() / "runs")
    evaluator_now = evaluator_identity()

    for r in runs:
        r["evidence"] = assess_run_evidence(r, cohort, evaluator_now)
        r["benchmark_evidence"] = assess_run_evidence(
            r, cohort, evaluator_now, artifact="benchmark"
        )

    run_rows = []
    for r in runs:
        m = r["manifest"] or {}
        ev = r["evaluation"]
        tr = m.get("training") or {}
        run_rows.append({
            "run_id": m.get("run_id") or r["run_id"],
            "status": m.get("status"),
            "evidence": r["evidence"]["state"],
            "evidence_reason": r["evidence"]["reason"],
            "benchmark_evidence": r["benchmark_evidence"]["state"],
            "system": (m.get("experiment") or {}).get("system"),
            "architecture": (m.get("experiment") or {}).get("architecture"),
            "condition": (m.get("experiment") or {}).get("condition"),
            "seed": (m.get("experiment") or {}).get("seed"),
            "lambda_pde": (m.get("experiment") or {}).get("lambda_pde"),
            "best_val_loss": tr.get("best_val_loss"),
            "best_epoch": tr.get("best_epoch"),
            "last_epoch": tr.get("last_epoch"),
            "epochs_completed": tr.get("epochs_completed"),
            "train_walltime_s": tr.get("walltime_seconds"),
            "params_total": (m.get("parameter_counts") or {}).get("total"),
            "params_complex": (m.get("parameter_counts") or {}).get("complex"),
            "peak_cuda_gib": m.get("peak_cuda_allocated_gib"),
            "peak_host_rss_gib": m.get("peak_host_rss_gib"),
            "rollout_nrmse_mean": (ev or {}).get("nrmse", {}).get("mean"),
            "rollout_nrmse_median": (ev or {}).get("nrmse", {}).get("median"),
            "one_step_nrmse_mean": (ev or {}).get("nrmse_one_step_teacher_forced", {}).get("mean"),
            "one_step_nrmse_aggregation": (ev or {}).get(
                "nrmse_one_step_teacher_forced", {}
            ).get("aggregation"),
            "persistence_nrmse_mean": (ev or {}).get("persistence_baseline_nrmse", {}).get("mean"),
            "numerical_stepping_nrmse_mean": (ev or {}).get(
                "numerical_stepping_baseline_nrmse", {}
            ).get("mean"),
            "residual_rms_implicit_euler_first_order": (ev or {}).get(
                "residual_rms_implicit_euler_first_order"
            ),
            "reference_residual_rms_implicit_euler_first_order": (ev or {}).get(
                "reference_residual_rms_implicit_euler_first_order"
            ),
            "residual_rms_centred_first_derivative": (ev or {}).get(
                "residual_rms_centred_first_derivative"
            ),
            "reference_residual_rms_centred_first_derivative": (ev or {}).get(
                "reference_residual_rms_centred_first_derivative"
            ),
            "throughput_ratio": (r["benchmark"] or {}).get("throughput_ratio"),
            "memory_reduction_percent": (r["benchmark"] or {}).get("memory_reduction_percent"),
            "test_dataset_sha256": (ev or {}).get("test_dataset_sha256"),
        })

    target_rows = build_target_table(runs, targets, cohort, evaluator_now)

    # ---- markdown summary ----
    lines: list[str] = []
    lines.append("# PDELab Results\n")
    lines.append(
        f"Generated from artifacts under `runs/`. "
        f"Artifact timestamp: {_artifact_timestamp(runs)}\n"
    )

    lines.append("\n## Declared targets\n")
    lines.append(
        "Targets are *proposals* declared in `configs/targets.yaml`, not "
        "accomplishments. A target is evaluated only against the **declared "
        "cohort**; a run that does not match the cohort identity is excluded "
        "with a reason and is never substituted. Status is one of "
        + " / ".join(TARGET_STATUSES)
        + ".\n"
    )
    if cohort:
        lines.append(
            f"Cohort `{cohort.get('cohort_id')}` "
            f"(declared in `{DEFAULT_COHORT_PATH}`).\n"
        )
    else:
        lines.append(
            f"No cohort declaration was found at `{DEFAULT_COHORT_PATH}`; "
            "no target can be met.\n"
        )
    lines.append(
        "| target | metric | threshold | status | measured | units | run_id |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for t in target_rows:
        lines.append(
            f"| {t['target']} | {t['metric']} | {t['comparison']} {_fmt(t['threshold'])} "
            f"| **{t['status']}** | {_fmt(t['measured'])} | {t['units']} "
            f"| {t['run_id'] or '-'} |"
        )
    for t in target_rows:
        if t.get("note"):
            lines.append(f"\n- `{t['target']}`: {t['note']}")

    lines.append("\n\n## Runs\n")
    if run_rows:
        headers = list(run_rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in run_rows:
            lines.append("| " + " | ".join(_fmt(v) for v in row.values()) + " |")
    else:
        lines.append("_No runs found._")

    lines.append("\n## Protocol notes\n")
    lines.append(
        "- Data metric: per-trajectory relative RMS error over the predicted "
        "future snapshots only; the copied initial frame is excluded.\n"
        "- One-step metric: the mean is taken over *trajectories* (the "
        "per-window values are averaged inside a trajectory first); the "
        "per-window summary is reported separately and labelled as such.\n"
        "- Residual metric: physical-unit RMS PDE residual over the declared "
        "space/time domain, with an identical evaluator, grid, time spacing "
        "and aggregation for every condition. The reference solution's own "
        "residual under the same evaluator is reported alongside.\n"
        "- Two residual evaluators are declared and never mixed: the primary "
        "implicit-Euler (first order) residual and the centred first-derivative "
        "residual. Both are evaluated on the *same* support - the interior "
        "snapshot indices 1 .. T-2 - so their values are comparable.\n"
        "- The primary evaluator's truncation error is measured, not asserted: "
        "each artifact carries a temporal refinement probe of the reference "
        "solution's own residual against dt.\n"
        "- Baselines: persistence (parameter free) and a low-cost numerical "
        "stepping baseline on the same grid and horizon (parameter informed: "
        "it needs the PDE and its parameters). Both are reported with their "
        "accuracy and their runtime.\n"
        "- Every artifact is bound to the SHA-256 of the checkpoint, the "
        "dataset split, the normalisation, the evaluator implementation, the "
        "resolved settings and the protocol; artifacts are written atomically "
        "and hashed afterwards.\n"
        "- Inference comparisons use the same checkpoint, grid, horizon, batch "
        "size and output contract; each variant is measured in a separate "
        "process, on the declared reference model.\n"
    )

    md_text = "\n".join(lines) + "\n"
    md_path = out_dir / "RESULTS.md"
    write_text_atomic(md_path, md_text)
    docs_path = project_root() / "docs" / "RESULTS.md"
    if write_docs:
        # The same generated table is mirrored into docs/ so that the
        # documentation never contains hand-written numbers.
        docs_dir = project_root() / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomic(docs_path, md_text)

    json_path = out_dir / "results.json"
    write_json_atomic(
        json_path,
        {
            "generated_from_artifacts_utc": _artifact_timestamp(runs),
            "cohort": cohort.get("cohort_id"),
            "evaluator_identity": evaluator_now,
            "n_runs": len(run_rows),
            "n_runs_completed": sum(1 for r in run_rows if r["status"] == "completed"),
            "n_evidence_valid": sum(1 for r in run_rows if r["evidence"] == "valid"),
            "runs": run_rows,
            "targets": target_rows,
        },
    )

    plots = []
    if make_plots:
        from .plots import generate_all

        try:
            plot_result = generate_all()
            plots = plot_result.get("plots", [])
        except Exception as exc:
            plots = [f"plot generation failed: {type(exc).__name__}: {exc}"]

    return {
        "status": "completed",
        "cohort": cohort.get("cohort_id"),
        "n_runs_found": len(run_rows),
        "n_runs_completed": sum(1 for r in run_rows if r["status"] == "completed"),
        "n_evidence_valid": sum(1 for r in run_rows if r["evidence"] == "valid"),
        "n_targets": len(target_rows),
        "targets_met": sum(1 for t in target_rows if t["status"] == "met"),
        "targets_missed": sum(1 for t in target_rows if t["status"] == "missed"),
        "targets_not_run": sum(1 for t in target_rows if t["status"] == "not_run"),
        "targets_incomplete_cohort": sum(
            1 for t in target_rows if t["status"] == "incomplete_cohort"
        ),
        "targets_not_comparable": sum(
            1 for t in target_rows if t["status"] == "not_comparable"
        ),
        "targets_stale": sum(1 for t in target_rows if t["status"] == "stale"),
        "targets_invalid_protocol": sum(
            1 for t in target_rows if t["status"] == "invalid_protocol"
        ),
        "results_markdown": str(md_path),
        "results_json": str(json_path),
        "docs_results_markdown": str(docs_path),
        "plots": plots,
    }
