# campaigns/performance

Evidence and fixtures of the performance layer of PDELab.  Everything
here is a *fixture or a probe*: it produces numbers that can be quoted,
and it does not change the behaviour of the library.

| file | what it is |
|---|---|
| `benchmark_protocol.md` | the **frozen** protocol of the inference benchmark, including the acceptance tolerance that was declared before any candidate was measured |
| `test_benchmark_correctness.py` | fixture: the element-wise accuracy check rejects a **sign flip** and a **spatial permutation** that the old scalar checksum accepted; runnable with pytest and as a script |
| `check_benchmark_identity.py` | runs the report's own evidence assessment on a benchmark artifact, so that the identity field names are verified against the code that reads them |
| `check_architecture_change.py` | shows that the old FNO2D checkpoints do **not** load into the repaired model (they are a different architecture) and that the benchmark artifact produced before the identity-bound protocol is refused by the report |
| `benchmark_smoke.py` | a bounded (one block, one iteration) GPU smoke of the whole benchmark path on a *copy* of an existing run directory; it never overwrites an artifact under `runs/` |
| `check_project_lease.py` | checks the project lease of the heavy entrypoints: refusal while another PDELab job holds it, mutual exclusion of two plain locks, and the joining of a nested lock |
| `fno2d_spectral_probe.py` | measures the spectral coverage of the 2D FNO spectral convolution (the identity-weight operator against its own input, a single cosine, the Hermitian spectrum); it runs against the old and the repaired implementation |
| `fno2d_parameter_counts.py` | the parameter counts of the FNO2D architecture before and after the spectral repair |

Run them with the project virtual environment, from the project root:

```bash
cd /home/atulp/projects/pdelab
.venv/bin/python campaigns/performance/test_benchmark_correctness.py
.venv/bin/python campaigns/performance/check_project_lease.py
.venv/bin/python campaigns/performance/fno2d_spectral_probe.py
.venv/bin/python campaigns/performance/fno2d_parameter_counts.py
.venv/bin/python -m pytest campaigns/performance/test_benchmark_correctness.py -q
```

The GPU smoke needs an idle GPU and writes only into
`/tmp/pdelab_perf/`:

```bash
nvidia-smi
.venv/bin/python campaigns/performance/benchmark_smoke.py
.venv/bin/python campaigns/performance/check_benchmark_identity.py /tmp/pdelab_perf/bench_runs
```
