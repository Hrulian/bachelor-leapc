"""Planner using the FULL cart+pendulum model with a learnable LINEAR COST
PERTURBATION on the first input (my_acados_ocp_du0.py).

Same as my_planner.py (full 4-state model, full observation, no obs slicing),
except that no reference is learnable at all: the actor outputs ``d_u0``, the
gradient perturbation of Gros & Zanon (2021), Eq. (11), which shifts only the
stage-0 input reference. Lets us compare "learn a reference the MPC tracks"
against "perturb the MPC's cost gradient" with an identical dynamics model.

The action bound scales with Fmax (|d| <= r * Fmax), so changing Fmax in the
config automatically rescales the parameter space — see `d_u0_bound`.
"""

from dataclasses import dataclass
from pathlib import Path

from bachelor.acados_cartpole.my_acados_ocp_du0 import (
    CartPoleAcadosCostType,
    CartPoleAcadosParamInterface,
    create_custom_cartpole_params,
    export_parametric_ocp,
)
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.torch import AcadosDiffMpcTorch


@dataclass(kw_only=True)
class GradPertPlannerConfig:
    """Configuration for the full-model / cost-gradient-perturbation planner.

    Same high-level knobs as CartPolePlannerConfig in my_planner.py (full model),
    so the two can be swapped in the training script. See my_planner.py for the
    meaning of each field.
    """

    N_horizon: int = 8  # chosen so that the avg MPC call is less than 15ms
    T_horizon: float = 0.45  # chosen so that the avg MPC call is less than 15ms
    Fmax: float = 20.0  # chosen so that v is below 2 m/s
    x_threshold: float = 0.37

    cost_type: CartPoleAcadosCostType = "NONLINEAR_LS"
    param_interface: CartPoleAcadosParamInterface = "global"

    # Half-width of the d_u0 action box. None = the textbook bound r * Fmax
    # (= 2.0 with r = 0.1, Fmax = 20). That bound is measurably far too small:
    # sweeping it moves the applied force by only ~0.001 - 1.4 N. Set this to
    # ~1e2 - 1e3 to give the actor authority comparable to the reference-based
    # planners. See `d_u0_bound` in my_acados_ocp_du0.py.
    d_max: float | None = None


class GradPertPlanner(AcadosPlanner):
    """Acados planner: full cart+pendulum dynamics, learnable cost perturbation d_u0.

    Uses the full [x, theta, v, thetadot] state, so — like CartPolePlanner and
    unlike CartOnlyPlanner — the observation is passed through unchanged.
    """

    cfg: GradPertPlannerConfig

    def __init__(
        self,
        cfg: GradPertPlannerConfig | None = None,
        params: list[AcadosParameter] | None = None,
        export_directory: Path | None = None,
    ):
        self.cfg = GradPertPlannerConfig() if cfg is None else cfg
        params = (
            create_custom_cartpole_params(
                param_interface=self.cfg.param_interface,
                N_horizon=self.cfg.N_horizon,
                Fmax=self.cfg.Fmax,
                d_max=self.cfg.d_max,
            )
            if params is None
            else params
        )

        param_manager = AcadosParameterManager(parameters=params, N_horizon=self.cfg.N_horizon)

        ocp = export_parametric_ocp(
            param_manager=param_manager,
            cost_type=self.cfg.cost_type,
            name="cartpole_du0",
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            Fmax=self.cfg.Fmax,
            x_threshold=self.cfg.x_threshold,
        )

        diff_mpc = AcadosDiffMpcTorch(ocp, export_directory=export_directory)
        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)
