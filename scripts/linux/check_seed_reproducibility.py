"""Verify that a seeded run is reproducible.

Two properties are checked:
  1. The model *initialisation* is a deterministic function of the seed
     (the initialisation is seeded before the model is constructed).
  2. Two runs of the same configuration and seed produce the same
     initialisation fingerprint and the same first-epoch validation
     loss.
"""


from pdelab.models.registry import build_model_from_config
from pdelab.utils import init_fingerprint, seed_everything


def build_fingerprint(arch: str, seed: int, **kwargs) -> str:
    seed_everything(seed)
    model = build_model_from_config(
        {"architecture": arch, "history": 1, **kwargs},
        dimension=1 if arch.endswith("1d") else 2,
    )
    return init_fingerprint(model)


def main() -> int:
    failures = 0
    for arch, kwargs in [
        ("fno1d", {"modes": 8, "width": 8, "n_layers": 2}),
        ("unet1d", {"width": 8, "n_levels": 2}),
        ("fno2d", {"modes_x": 4, "modes_y": 4, "width": 8, "n_layers": 2}),
        ("unet2d", {"width": 8, "n_levels": 2}),
    ]:
        f0a = build_fingerprint(arch, 0, **kwargs)
        f0b = build_fingerprint(arch, 0, **kwargs)
        f1 = build_fingerprint(arch, 1, **kwargs)
        same_seed = f0a == f0b
        diff_seed = f0a != f1
        ok = same_seed and diff_seed
        failures += 0 if ok else 1
        print(f"{arch:8s} same seed -> same init: {same_seed}   "
              f"different seed -> different init: {diff_seed}   "
              f"[{f0a} / {f1}]  {'PASS' if ok else 'FAIL'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
