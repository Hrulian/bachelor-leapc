"""Registry of planners to compare against each other in one wandb project.

Each entry maps a name to a builder that returns a ready-to-use planner. The
simulation environment always passes the FULL [x, theta, v, thetadot] state, so
planners using a reduced model adapt internally (e.g. the cart-only planner
slices the observation down to [x, v]). This keeps the env identical across runs
so different OCPs / parameter interfaces are compared fairly.

To test another interface or OCP, add a builder + registry entry here; it then
becomes selectable via --planner <name> in sim_sac_zop.py and run_parallel.py.
"""

from pathlib import Path


def _build_full(export_directory: Path | None = None):
    """Full cart + pendulum model, global parameter interface (my_planner.py)."""
    from bachelor.acados_cartpole.my_planner import CartPolePlannerConfig, CartPolePlanner

    cfg = CartPolePlannerConfig(param_interface="global")
    return CartPolePlanner(cfg, export_directory=export_directory)


def _build_cart(export_directory: Path | None = None):
    """Simplified cart-only model, global interface (my_planner_cart.py)."""
    from bachelor.acados_cartpole.my_planner_cart import CartOnlyPlannerConfig, CartOnlyPlanner

    cfg = CartOnlyPlannerConfig(param_interface="global")
    return CartOnlyPlanner(cfg, export_directory=export_directory)


def _build_fullcart(export_directory: Path | None = None):
    """Full cart+pendulum model, but the learnable reference is the cart position
    instead of the pole angle, global interface (my_planner_fullcart.py)."""
    from bachelor.acados_cartpole.my_planner_fullcart import (
        FullCartRefPlannerConfig,
        FullCartRefPlanner,
    )

    cfg = FullCartRefPlannerConfig(param_interface="global")
    return FullCartRefPlanner(cfg, export_directory=export_directory)


PLANNER_REGISTRY = {
    "full": _build_full,
    "cart": _build_cart,
    "fullcart": _build_fullcart,
}


def build_planner(name: str, export_directory: Path | None = None):
    if name not in PLANNER_REGISTRY:
        raise ValueError(
            f"Unknown planner '{name}'. Available: {sorted(PLANNER_REGISTRY)}"
        )
    export_directory = Path(export_directory) if export_directory is not None else None
    return PLANNER_REGISTRY[name](export_directory)
