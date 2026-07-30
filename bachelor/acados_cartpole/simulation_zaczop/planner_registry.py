"""Compatibility shim: the planner registry now lives one level up.

The real-hardware scripts and the sim scripts have to agree on which OCP variants
exist, so the registry was moved to `bachelor/acados_cartpole/my_planner_registry.py`
and this module just re-exports it. Everything importing `build_planner` or
`PLANNER_REGISTRY` from here keeps working unchanged, and now also sees the
`tunable` variant.

Add new planners in my_planner_registry.py, not here.
"""

from bachelor.acados_cartpole.my_planner_registry import (  # noqa: F401
    PLANNER_NAMES,
    PLANNER_REGISTRY,
    build_planner,
    canonical_name,
    make_planner,
    planner_config_dict,
)

__all__ = [
    "PLANNER_NAMES",
    "PLANNER_REGISTRY",
    "build_planner",
    "canonical_name",
    "make_planner",
    "planner_config_dict",
]
