"""Planner using the FULL cart+pendulum model but with the CART POSITION as the
learnable reference (my_acados_ocp_fullcart.py).

Same as my_planner.py (full 4-state model, full observation, no obs slicing),
except the learnable parameter is xref0 (cart position) instead of xref1 (pole
angle). Lets us compare "learn where to put the cart" against "learn the pole
reference" while the MPC dynamics model is identical in both.
"""

from dataclasses import dataclass
from pathlib import Path

from bachelor.acados_cartpole.my_acados_ocp_fullcart import (
    CartPoleAcadosCostType,
    CartPoleAcadosParamInterface,
    create_custom_cartpole_params,
    export_parametric_ocp,
)
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.torch import AcadosDiffMpcTorch


@dataclass(kw_only=True)
class FullCartRefPlannerConfig:
    """Configuration for the full-model / cart-position-reference planner.

    Same high-level knobs as CartPolePlannerConfig in my_planner.py (full model),
    so the two can be swapped in the training script. See my_planner.py for the
    meaning of each field.
    """

    N_horizon: int = 5  # chosen so that the avg MPC call is less than 15ms
    T_horizon: float = 0.25  # chosen so that the avg MPC call is less than 15ms
    Fmax: float = 20.0  # chosen so that v is below 2 m/s
    x_threshold: float = 0.4

    cost_type: CartPoleAcadosCostType = "NONLINEAR_LS"
    param_interface: CartPoleAcadosParamInterface = "global"


class FullCartRefPlanner(AcadosPlanner):
    """Acados planner: full cart+pendulum dynamics, learnable cart-position reference.

    Uses the full [x, theta, v, thetadot] state, so — unlike CartOnlyPlanner —
    the observation is passed through unchanged (no slicing).
    """

    cfg: FullCartRefPlannerConfig

    def __init__(
        self,
        cfg: FullCartRefPlannerConfig | None = None,
        params: list[AcadosParameter] | None = None,
        export_directory: Path | None = None,
    ):
        self.cfg = FullCartRefPlannerConfig() if cfg is None else cfg
        params = (
            create_custom_cartpole_params(
                param_interface=self.cfg.param_interface,
                N_horizon=self.cfg.N_horizon,
            )
            if params is None
            else params
        )

        param_manager = AcadosParameterManager(parameters=params, N_horizon=self.cfg.N_horizon)

        ocp = export_parametric_ocp(
            param_manager=param_manager,
            cost_type=self.cfg.cost_type,
            name="cartpole_fullcart",
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            Fmax=self.cfg.Fmax,
            x_threshold=self.cfg.x_threshold,
        )

        diff_mpc = AcadosDiffMpcTorch(ocp, export_directory=export_directory)
        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)
