"""Report the parameter counts of every architecture variant.

Used to choose the near parameter-matched U-Net width and to document
the counts in the report.
"""

from pdelab.models.base import count_parameters
from pdelab.models.registry import architecture_defaults, build_model

VARIANTS = [
    ("fno1d", 1, 16, None),
    ("unet1d", 1, 16, None),
    ("unet1d", 1, 6, None),
    ("unet1d", 1, 4, None),
    ("fno2d", 2, 16, None),
    ("unet2d", 2, 16, None),
    ("unet2d", 2, 6, None),
    ("unet2d", 2, 4, None),
]

print(f"{'arch':8s} {'width':>6s} {'total':>9s} {'complex':>9s} {'real':>8s}")
print("-" * 38)
for arch, dim, width, _ in VARIANTS:
    kwargs = dict(architecture_defaults(arch))
    if arch.startswith("unet"):
        kwargs["width"] = width
    model = build_model(arch, dimension=dim, **kwargs)
    c = count_parameters(model)
    print(f"{arch:8s} {width:>6d} {c['total']:>9d} {c['complex']:>9d} {c['real']:>8d}")
