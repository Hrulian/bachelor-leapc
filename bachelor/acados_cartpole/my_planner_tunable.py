"""Runtime-tunable planner wrapping my_acados_ocp_tunable.py.

One config object exposes EVERYTHING that defines the MPC — model parameters
(masses, length, gravity), cost weights, horizon, bounds and which constraints
are active — so a search script can build arbitrary variants without editing
code. Defaults reproduce the matched, task-solving "full" MPC.

    from bachelor.acados_cartpole.my_planner_tunable import (
        TunablePlanner, TunablePlannerConfig,
    )
    cfg = TunablePlannerConfig(name="weak_0", M=0.17444 * 8)  # 8x too-heavy cart
    planner = TunablePlanner(cfg, export_directory=...)
"""

from dataclasses import dataclass
from pathlib import Path

from bachelor.acados_cartpole.my_acados_ocp_tunable import (
    CartPoleAcadosCostType,
    CartPoleAcadosParamInterface,
    TunableModel,
    create_custom_cartpole_params,
    export_parametric_ocp,
)
from leap_c.ocp.acados.parameters import AcadosParameterManager
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.torch import AcadosDiffMpcTorch


@dataclass(kw_only=True)
class TunablePlannerConfig:
    """Every knob of the MPC in one place. Defaults == matched "full" planner
    (the one that already solves the task)."""

    # --- unique identity (must differ between variants built in one process) ---
    name: str = "cartpole_tunable"

    # --- horizon / bounds / solver ---
    N_horizon: int = 8
    T_horizon: float = 0.45
    Fmax: float = 20.0
    x_threshold: float = 0.4
    cost_type: CartPoleAcadosCostType = "NONLINEAR_LS"
    param_interface: CartPoleAcadosParamInterface = "global"

    # --- constraint toggles ---
    use_force_constraint: bool = True
    use_position_constraint: bool = True

    # --- MPC internal model (perturb these for model-plant mismatch) ---
    M: float = 0.17444
    m: float = 0.016
    g: float = 9.81
    l: float = 0.36
    Ip: float = 0.0001728

    # --- cost weights ---
    q_diag: tuple = (1e3, 1e3, 1.0, 1.0)  # on [x, theta, v, thetadot]
    r_diag: tuple = (1e-1,)               # on force

    def model(self) -> TunableModel:
        return TunableModel(
            M=self.M, m=self.m, g=self.g, l=self.l, Ip=self.Ip,
            q_diag=self.q_diag, r_diag=self.r_diag,
        )


class TunablePlanner(AcadosPlanner):
    """Full cart+pendulum planner whose entire definition comes from a
    TunablePlannerConfig. Uses the full [x, theta, v, thetadot] observation
    (no slicing), so it is a drop-in for the "full" planner."""

    cfg: TunablePlannerConfig

    def __init__(
        self,
        cfg: TunablePlannerConfig | None = None,
        export_directory: Path | None = None,
    ):
        self.cfg = TunablePlannerConfig() if cfg is None else cfg

        params = create_custom_cartpole_params(
            param_interface=self.cfg.param_interface,
            N_horizon=self.cfg.N_horizon,
            model=self.cfg.model(),
        )
        param_manager = AcadosParameterManager(parameters=params, N_horizon=self.cfg.N_horizon)

        ocp = export_parametric_ocp(
            param_manager=param_manager,
            cost_type=self.cfg.cost_type,
            name=self.cfg.name,
            Fmax=self.cfg.Fmax,
            x_threshold=self.cfg.x_threshold,
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            use_force_constraint=self.cfg.use_force_constraint,
            use_position_constraint=self.cfg.use_position_constraint,
        )

        diff_mpc = AcadosDiffMpcTorch(ocp, export_directory=export_directory)
        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)
