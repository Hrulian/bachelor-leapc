"""Single registry of every MPC planner / OCP formulation, shared by sim and real.

One place to list the OCP variants, one short name to switch between them:

    from bachelor.acados_cartpole.my_planner_registry import make_planner

    planner, cfg = make_planner("full")                       # + its config
    planner, cfg = make_planner("cart", N_horizon=8)          # with overrides
    planner       = build_planner("full", export_directory)   # planner only

Which name maps to which file:

    name        module                    class                 model / learnable
    ---------------------------------------------------------------------------
    full        my_planner.py             CartPolePlanner       full cart+pole,
                                                                ref = pole angle
    cart        my_planner_cart.py        CartOnlyPlanner       cart only [x, v]
    fullcart    my_planner_fullcart.py    FullCartRefPlanner    full cart+pole,
                                                                ref = cart position
    du0         my_planner_du0.py         GradPertPlanner       full cart+pole,
                                                                no learnable ref,
                                                                cost perturbation
                                                                d^T u_0 instead
    tunable     my_planner_tunable.py     TunablePlanner        full cart+pole,
                                                                every knob exposed
    gz          my_planner_gz.py          GZPlanner             full cart+pole,
                                                                ref = pole angle
                                                                LEARNED, plus a
                                                                SAMPLED d_u0
                                                                disturbance for
                                                                exploration
                                                                (Gros & Zanon)

`full` is the plain `my_planner.py` planner - the default everywhere, and the one
real_sac_zop.py used before this registry existed. It is also reachable under the
alias `my_planner` (see `_ALIASES`) so either name works.

To add a variant, drop a `my_planner_<x>.py` next to this file and add one line to
`_REGISTRY`. It then becomes selectable via `--planner <x>` in the sim scripts and
via `PLANNER=<x>` in real_sac_zop.py, with no other change.

Imports are lazy: only the requested variant's module is loaded, so a broken or
half-finished OCP formulation cannot break the runs that don't use it.

NOTE: the variants do not share an OCP formulation, so a policy trained on one is
meaningless on another even where the tensor shapes happen to line up. The
training scripts therefore keep a separate checkpoint directory per planner name.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

# canonical name -> (module path, config class, planner class)
_REGISTRY: dict[str, tuple[str, str, str]] = {
    "full": (
        "bachelor.acados_cartpole.my_planner",
        "CartPolePlannerConfig",
        "CartPolePlanner",
    ),
    "cart": (
        "bachelor.acados_cartpole.my_planner_cart",
        "CartOnlyPlannerConfig",
        "CartOnlyPlanner",
    ),
    "fullcart": (
        "bachelor.acados_cartpole.my_planner_fullcart",
        "FullCartRefPlannerConfig",
        "FullCartRefPlanner",
    ),
    "du0": (
        "bachelor.acados_cartpole.my_planner_du0",
        "GradPertPlannerConfig",
        "GradPertPlanner",
    ),
    "tunable": (
        "bachelor.acados_cartpole.my_planner_tunable",
        "TunablePlannerConfig",
        "TunablePlanner",
    ),
    "gz": (
        "bachelor.acados_cartpole.my_planner_gz",
        "GZPlannerConfig",
        "GZPlanner",
    ),
}

# alternative spellings -> canonical name. Kept out of PLANNER_NAMES so argparse
# `choices` and `--help` stay short, but accepted everywhere a name is taken.
_ALIASES: dict[str, str] = {
    "my_planner": "full",
    "my_planner_cart": "cart",
    "my_planner_fullcart": "fullcart",
    "my_planner_du0": "du0",
    "gradpert": "du0",
    "d_u0": "du0",
    "my_planner_tunable": "tunable",
    "cartpole": "full",
}

PLANNER_NAMES: tuple[str, ...] = tuple(_REGISTRY)


def canonical_name(name: str) -> str:
    """Resolve an alias to its canonical registry name, validating it."""
    resolved = _ALIASES.get(name, name)
    if resolved not in _REGISTRY:
        raise KeyError(
            f"Unknown planner '{name}'. Available: {', '.join(PLANNER_NAMES)}"
            f" (aliases: {', '.join(sorted(_ALIASES))})"
        )
    return resolved


def _resolve(name: str):
    """Import the requested variant, returning its (config class, planner class)."""
    module_path, cfg_cls_name, planner_cls_name = _REGISTRY[canonical_name(name)]
    module = importlib.import_module(module_path)
    return getattr(module, cfg_cls_name), getattr(module, planner_cls_name)


def make_planner(
    name: str,
    export_directory: Path | None = None,
    **cfg_overrides: Any,
):
    """Build one of the registered planners, returning it together with its config.

    Args:
        name: A registry name or alias, e.g. "full", "cart", "my_planner".
        export_directory: Where acados writes its generated C code. `None` uses the
            planner's own default; each variant exports under a distinct solver
            name, so the defaults do not collide.
        **cfg_overrides: Passed to the variant's config dataclass, e.g.
            `make_planner("tunable", M=0.17444 * 8, name="weak_0")`.

    Returns:
        (planner, cfg) - the built planner and the config it was built from.
    """
    cfg_cls, planner_cls = _resolve(name)
    cfg = cfg_cls(**cfg_overrides)
    export_directory = Path(export_directory) if export_directory is not None else None
    planner = planner_cls(cfg, export_directory=export_directory)
    return planner, cfg


def build_planner(name: str, export_directory: Path | None = None):
    """Build a planner and return only it (the signature the sim scripts use)."""
    planner, _cfg = make_planner(name, export_directory)
    return planner


def planner_config_dict(name: str, cfg: Any) -> dict[str, Any]:
    """Flatten a planner config into wandb-loggable scalars.

    Prefixed with `planner/` so the OCP settings stay grouped in the wandb config
    and every run records which formulation produced it.
    """
    out: dict[str, Any] = {"planner/name": canonical_name(name)}
    if is_dataclass(cfg):
        for key, value in asdict(cfg).items():
            out[f"planner/{key}"] = list(value) if isinstance(value, tuple) else value
    return out


# Backwards-compatible mapping name -> builder callable. The sim scripts use it as
# `choices=sorted(PLANNER_REGISTRY)` for argparse, so it holds canonical names only.
PLANNER_REGISTRY: dict[str, Any] = {
    _name: (lambda export_directory=None, _n=_name: build_planner(_n, export_directory))
    for _name in _REGISTRY
}
