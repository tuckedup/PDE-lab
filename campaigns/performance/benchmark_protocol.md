# Inference benchmark protocol (frozen)

This is the protocol of the matched inference benchmark
(`src/pdelab/benchmarks/inference.py`).  It is **frozen**: the
acceptance tolerance below was declared *before* any candidate was
measured and it is never loosened.  A candidate that fails it is
rejected, and the speed and memory numbers of a rejected candidate do
not count towards a target.

| field | value |
|---|---|
| benchmark protocol id | `pdelab-inference-benchmark/1` |
| cohort protocol id (recorded in `identity.protocol.id`) | `pdelab-free-rollout/2` |
| tolerance frozen (UTC) | 2026-10-21T00:00:00Z |
| accuracy split | `val` (the development split; the test split stays reserved for the reported metric) |
| accuracy workload | several batches of development trajectories, the **entire** rollout horizon |
| isolation | every variant is measured in a **separate process** |
| workload | the same checkpoint, grid, horizon, batch size and output contract for every variant |

## Acceptance tolerance (frozen)

| check | limit |
|---|---|
| `max_abs_error` | 5.0e-2 (physical units of the field) |
| `relative_l2` | 1.0e-2 |
| `max_rollout_nrmse_delta` | 1.0e-2 (absolute change of the rollout nRMSE) |
| `max_nonfinite_count` | 0 |

The numbers live in `AccuracyTolerance` /
`ACCURACY_TOLERANCE` in `src/pdelab/benchmarks/inference.py` and are
copied into every artifact (`accuracy.tolerance`,
`identity.protocol.benchmark.tolerance`).

## What the accuracy check compares

The **full output tensors** of the FP32 baseline and of the variant,
on **identical inputs** (the same input file is handed to every
variant process), over the entire rollout and over several development
batches:

* the maximum absolute error,
* the relative L2 error,
* the counts of non-finite values on both sides,
* the difference at **every** time step of the rollout,
* the change of the rollout nRMSE against the reference solution.

The scalar `output_checksum` (the mean absolute value of the rollout)
is kept only as a supplementary **identity** diagnostic - it says that
two processes produced the same numbers.  It is never an accuracy
proof: a sign flip and a spatial permutation have exactly the same
mean absolute value, and both are rejected by the element-wise check.
`campaigns/performance/test_benchmark_correctness.py` is the fixture
that proves it.

## Content identity

Every artifact records the identity the report binds to:

| field | content |
|---|---|
| `identity.evaluator.sha256` | the content identity of the evaluation implementation |
| `identity.evaluator.id` | `pdelab-evaluation/2` |
| `identity.protocol.id` | the cohort's declared protocol id |
| `identity.protocol.benchmark.id` | the benchmark-specific protocol id |
| `identity.resolved_settings_sha256` | the digest of the resolved benchmark settings |
| `identity.checkpoint_sha256` | the digest of the checkpoint that was benchmarked |
| `identity.dataset.path` / `.sha256` / `.split` | the evaluated dataset split |
| `identity.normalization.sha256` | the digest of the normalisation constants |

The artifact is written atomically and its digest is stored next to it
in `benchmark_matched.json.sha256`, so that a truncated or modified
artifact is detected rather than silently parsed.

`identity.protocol.id` carries the cohort's protocol id because the
report binds every artifact of the cohort to that single id; the
benchmark-specific protocol id is recorded next to it in
`identity.protocol.benchmark.id`, together with the frozen tolerance.
