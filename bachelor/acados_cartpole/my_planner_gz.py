"""Planner for the Gros & Zanon split: LEARN the reference, SAMPLE the disturbance.

Same full 4-state model as my_planner.py, and the same learnable pole-angle
reference xref1 — but the OCP additionally exposes ``d_u0``, the gradient
disturbance of Gros & Zanon (arXiv:1906.04057, Eq. (45)), which the training
script fills with SAMPLED NOISE rather than learning it.

Contrast with the two neighbours:

    my_planner.py       learns xref1, explores by sampling in xref1-space
    my_planner_du0.py   learns d_u0, references frozen  (inverse of the paper)
    my_planner_gz.py    learns xref1, explores via sampled d_u0  (the paper)

The parameter vector handed to the solver is ``[xref1, d_u0]`` in that order.
"""

from dataclasses import dataclass
from pathlib import Path

from bachelor.acados_cartpole.my_acados_ocp_gz import (
    CartPoleAcadosCostType,
    CartPoleAcadosParamInterface,
    create_custom_cartpole_params,
    export_parametric_ocp,
)
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.torch import AcadosDiffMpcTorch


@dataclass(kw_only=True)
class GZPlannerConfig:
    """Configuration for the Gros & Zanon-style planner.

    Same high-level knobs as CartPolePlannerConfig in my_planner.py, plus the
    width of the disturbance channel. See my_planner.py for the shared fields.
    """

    N_horizon: int = 8  # chosen so that the avg MPC call is less than 15ms
    T_horizon: float = 0.45  # chosen so that the avg MPC call is less than 15ms
    Fmax: float = 20.0  # chosen so that v is below 2 m/s
    x_threshold: float = 0.37

    cost_type: CartPoleAcadosCostType = "NONLINEAR_LS"
    param_interface: CartPoleAcadosParamInterface = "global"

    # Half-width of the d_u0 box. NOT an authority knob (the reference is) — it
    # only has to sit a few sigma above the exploration std so the sampled noise
    # is not clipped. The training script's --explore-sigma-d picks that std.
    d_max: float = 300.0


class GZPlanner(AcadosPlanner):
    """Full cart+pendulum dynamics, learnable xref1, sampled d_u0 disturbance."""

    cfg: GZPlannerConfig

    def __init__(
        self,
        cfg: GZPlannerConfig | None = None,
        params: list[AcadosParameter] | None = None,
        export_directory: Path | None = None,
    ):
        self.cfg = GZPlannerConfig() if cfg is None else cfg
        params = (
            create_custom_cartpole_params(
                param_interface=self.cfg.param_interface,
                N_horizon=self.cfg.N_horizon,
                d_max=self.cfg.d_max,
            )
            if params is None
            else params
        )

        param_manager = AcadosParameterManager(parameters=params, N_horizon=self.cfg.N_horizon)

        ocp = export_parametric_ocp(
            param_manager=param_manager,
            cost_type=self.cfg.cost_type,
            name="cartpole_gz",
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            Fmax=self.cfg.Fmax,
            x_threshold=self.cfg.x_threshold,
        )

        diff_mpc = AcadosDiffMpcTorch(ocp, export_directory=export_directory)
        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)
