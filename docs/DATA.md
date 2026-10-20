# Datasets, splits and normalization

## Storage format

**HDF5** (chunked, gzip-compressed, float32), one file per
(system, split), plus a JSON sidecar manifest.

```
data/<system>/
    <system>_train.h5     dataset "u"   (n_traj, n_times, *grid)  float32
                           dataset "ic_seeds" (n_traj,)  int64
    <system>_val.h5
    <system>_test.h5
    <system>_manifest.json
```

Trajectories are written **incrementally**, one at a time, so the
generator never holds the whole dataset in memory. Chunking is
`(1, n_times, *grid)`, i.e. one chunk per trajectory, which keeps the
read pattern of the training loop simple and sequential.

Measured sizes: `data/burgers` 4.1 MiB, `data/allen_cahn` 7.2 MiB,
against a 4 GiB budget.

## Splits

Splits are assigned at the **trajectory / initial-condition level**,
before any window is extracted. A window belongs to exactly one
trajectory, so multiple windows of one simulation can never land in
different splits. This is asserted by a unit test.

The split is deterministic:

```python
seed_sequence = np.random.SeedSequence(split_seed)   # 1000 (1D), 2000 (2D)
child_seeds   = seed_sequence.spawn(n_total)
ic_seed       = int(child_seeds[i].generate_state(1, dtype=np.uint32)[0])
rng           = np.random.default_rng(ic_seed)
```

Prototype sizes (per system): 64 train / 16 validation / 16 test
trajectories. This is a small engineering study, not a universal
sample-size requirement; a larger confirmatory dataset must be planned
and **frozen before** its test results are examined.

Seed families are kept separate and are never reused:

| purpose | seed |
|---|---|
| data generation / split | 1000 (burgers), 2000 (allen_cahn) |
| training | 0 (prototype), 0/1/2 (confirmatory) |
| evaluation / benchmark | deterministic (no sampling) |

## Manifest contents

Each trajectory record stores:

* the split and the trajectory index,
* the initial-condition seed,
* the PDE parameters in force,
* the observation grid and the observation times,
* the solver name and the solver version,
* the initial-condition spectrum (realised RMS, realised maximum, and
  every Fourier coefficient),
* the number of reference integration steps and the reference step size,
* a SHA-256 hash of the trajectory bytes.

The manifest also stores the SHA-256 of each HDF5 file, the Git
revision and dirty status, and the environment snapshot. The test-file
hash is carried into every evaluation artifact, so a metric can always
be tied to the exact data it was measured on.

## Restriction from the reference grid to the observation grid

The reference is computed on a finer grid than the observation grid
(256 -> 128 in 1D, 64 -> 32 in 2D). The restriction is **not** a naive
stride, which would alias the modes between the observation Nyquist and
the reference Nyquist onto the observation grid. Instead:

1. the field is transformed with the full FFT,
2. modes with `|k|` above the observation band-limit are zeroed with a
   symmetric mask,
3. the filtered field is then sampled at the observation points.

A unit test asserts that a band-limited field survives the restriction
unchanged and that a high-frequency mode is attenuated rather than
aliased.

> An early implementation of the 2D restriction used `rfft2` with a
> one-sided row truncation. `rfft2` retains **both** signs of the first
> (non-halved) axis, so the truncation deleted the negative-kx modes and
> silently halved every field. The unit test caught it. The current
> implementation uses the full FFT with a symmetric `|k|` mask.

## Normalization

Statistics are computed from the **training split only** and applied
unchanged to validation, test and every model condition:

```python
stats = compute_normalization_stats(train_path)   # mean, std over the train split
```

* No per-trajectory normalization.
* No per-test-trajectory normalization using the target solution.
* The same statistics are used for the data-only and the
  physics-regularized condition of a given system, so the comparison is
  not confounded.
* Model outputs are **denormalised before** the physics residual and
  before the data error are computed, so every reported error is in
  physical units.

A unit test asserts that a test split with far larger values does not
influence the training statistics.

## Reproducibility

The stored data reproduces a fresh solve to float32 storage precision:
re-solving a test trajectory from its recorded `ic_seed` reproduces the
stored field to a maximum absolute difference of 6e-8.
