"""Experimental planner for testing simplified / alternative OCP formulations.

This mirrors my_planner.py but binds the cart-only OCP (my_acados_ocp_cart.py):
only the cart is modelled, a force acts on it, the pendulum is removed and only
the cart position is learnable.

Key design point: the simulation environment always passes the *full* state
[x, theta, v, thetadot] so different planners/OCPs can be compared against each
other in the same wandb project. A planner that uses a reduced model is
responsible for slicing the full observation down to its own OCP state itself
(see CART_OBS_IDX / the forward override below).

Add further experimental planners here (e.g. other parameter interfaces or
reduced models) and register them in the training script's planner registry.
"""

from dataclasses import dataclass
from pathlib import Path

from bachelor.acados_cartpole.my_acados_ocp_cart import (
    CART_OBS_IDX,
    CartPoleAcadosCostType,
    CartPoleAcadosParamInterface,
    create_custom_cartpole_params,
    export_parametric_ocp,
)
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.torch import AcadosDiffMpcTorch


@dataclass(kw_only=True)
class CartOnlyPlannerConfig:
    """Configuration for the cart-only CartPole planner.

    Same high-level knobs as CartPolePlannerConfig in my_planner.py so the two
    can be swapped in the training script. See my_planner.py for the meaning of
    each field.
    """

    N_horizon: int = 5
    T_horizon: float = 0.25
    Fmax: float = 20.0
    x_threshold: float = 0.4

    cost_type: CartPoleAcadosCostType = "NONLINEAR_LS"
    param_interface: CartPoleAcadosParamInterface = "global"


class CartOnlyPlanner(AcadosPlanner):
    """Acados planner using the simplified cart-only model (no pendulum).

    The OCP state is only [pos, vel]. The environment still provides the full
    [x, theta, v, thetadot] observation, so forward() slices it down to the cart
    states before handing it to the solver as the initial state.
    """

    cfg: CartOnlyPlannerConfig

    def __init__(
        self,
        cfg: CartOnlyPlannerConfig | None = None,
        params: list[AcadosParameter] | None = None,
        export_directory: Path | None = None,
    ):
        self.cfg = CartOnlyPlannerConfig() if cfg is None else cfg
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
            name="cartpole_cart",
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            Fmax=self.cfg.Fmax,
            x_threshold=self.cfg.x_threshold,
        )

        diff_mpc = AcadosDiffMpcTorch(ocp, export_directory=export_directory)
        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)

    def forward(self, obs, action=None, param=None, ctx=None):
        """Slice the full [x, theta, v, thetadot] observation to the cart states
        [x, v] expected by the reduced OCP, then solve as usual."""
        return super().forward(obs[:, CART_OBS_IDX], action=action, param=param, ctx=ctx)
