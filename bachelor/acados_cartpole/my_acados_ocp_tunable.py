# Fully tunable full-model cart+pendulum OCP.
#
# Same 4-state nonlinear dynamics and cost structure as my_acados_ocp.py, but
# EVERYTHING is injectable at build time so a search script can spin up many
# variants: the MPC's internal model parameters (M, m, g, l, Ip), the cost
# weights (q_diag, r_diag), horizon (N, T), bounds (Fmax, x_threshold), and
# whether the force / position box constraints are active.
#
# Use case: create a *model-plant mismatch* — feed the MPC wrong masses/length
# etc. while the sim env keeps the true dynamics. The optimization stays
# well-posed (converges, constraints respected) but the controller plans for the
# wrong plant, so it no longer solves the swing-up. See mpc_param_search.py.

from dataclasses import dataclass, field
from typing import Literal

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosModel, AcadosOcp
from leap_c.examples.utils.casadi import integrate_erk4
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

CartPoleAcadosParamInterface = Literal["global", "stagewise"]
CartPoleAcadosCostType = Literal["EXTERNAL", "NONLINEAR_LS"]


@dataclass
class TunableModel:
    """The MPC's internal model + cost. Defaults == the *true* sim values
    (RealCartPoleSimConfig), i.e. a perfectly matched, task-solving MPC.
    Scale these away from the truth to weaken the controller."""

    # dynamics parameters (the knobs the search perturbs)
    M: float = 0.17444   # cart mass [kg]
    m: float = 0.016     # pole mass [kg]
    g: float = 9.81      # gravity [m/s^2]
    l: float = 0.36      # pole length [m]
    Ip: float = 0.0001728  # pole inertia (currently unused in f_expl, kept for parity)
    # cost weights on [x, theta, v, thetadot] and on the force
    q_diag: tuple = (1e3, 1e3, 1.0, 1.0)
    r_diag: tuple = (1e-1,)


def create_custom_cartpole_params(
    param_interface: CartPoleAcadosParamInterface,
    N_horizon: int = 50,
    model: TunableModel | None = None,
) -> list[AcadosParameter]:
    """Full-model parameter list, but with the dynamics/cost values taken from
    ``model`` instead of hard-coded. The learnable reference is the pole angle
    (xref1), same interface as my_acados_ocp.py, so this stays a drop-in planner."""
    model = TunableModel() if model is None else model
    q_diag = np.asarray(model.q_diag, dtype=np.float64)
    r_diag = np.asarray(model.r_diag, dtype=np.float64)
    return [
        AcadosParameter("M", default=np.array([model.M])),
        AcadosParameter("m", default=np.array([model.m])),
        AcadosParameter("g", default=np.array([model.g])),
        AcadosParameter("l", default=np.array([model.l])),
        AcadosParameter("Ip", default=np.array([model.Ip])),
        AcadosParameter("q_diag_sqrt", default=np.sqrt(q_diag)),
        AcadosParameter("r_diag_sqrt", default=np.sqrt(r_diag)),
        AcadosParameter("xref0", default=np.array([0.0]), interface="non-learnable"),
        AcadosParameter(
            "xref1",
            default=np.array([0.0]),
            space=gym.spaces.Box(low=np.array([-np.pi]), high=np.array([np.pi]), dtype=np.float64),
            interface="learnable",
            end_stages=list(range(N_horizon + 1)) if param_interface == "stagewise" else [],
        ),
        AcadosParameter("xref2", default=np.array([0.0]), interface="non-learnable"),
        AcadosParameter("xref3", default=np.array([0.0]), interface="non-learnable"),
        AcadosParameter("uref", default=np.array([0.0]), interface="non-learnable"),
    ]


def define_f_expl_expr(model: AcadosModel, param_manager: AcadosParameterManager) -> ca.SX:
    """Full nonlinear cart+pendulum dynamics — identical to my_acados_ocp.py."""
    M = param_manager.get("M")
    m = param_manager.get("m")
    g = param_manager.get("g")
    l = param_manager.get("l")  # noqa: E741

    theta = model.x[1]
    v = model.x[2]
    dtheta = model.x[3]
    F = model.u[0]

    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    denominator = M + m - m * cos_theta * cos_theta

    f_expl = ca.vertcat(
        v,
        dtheta,
        (-m * l * sin_theta * dtheta * dtheta + m * g * cos_theta * sin_theta + F) / denominator,
        (-m * l * cos_theta * sin_theta * dtheta * dtheta + F * cos_theta + (M + m) * g * sin_theta)
        / (l * denominator),
    )
    return f_expl  # type: ignore


def export_parametric_ocp(
    param_manager: AcadosParameterManager,
    cost_type: CartPoleAcadosCostType = "NONLINEAR_LS",
    name: str = "cartpole_tunable",
    Fmax: float = 20.0,
    x_threshold: float = 0.4,
    N_horizon: int = 8,
    T_horizon: float = 0.45,
    use_force_constraint: bool = True,
    use_position_constraint: bool = True,
) -> AcadosOcp:
    ocp = AcadosOcp()
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon
    param_manager.assign_to_ocp(ocp)
    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon

    ######## Model ########
    ocp.model.name = name
    ocp.dims.nx = 4
    ocp.dims.nu = 1
    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)

    p = ca.vertcat(
        param_manager.non_learnable_parameters.cat,
        param_manager.learnable_parameters.cat,
    )  # type:ignore
    f_expl = define_f_expl_expr(ocp.model, param_manager)
    ocp.model.disc_dyn_expr = integrate_erk4(f_expl=f_expl, x=ocp.model.x, u=ocp.model.u, p=p, dt=dt)

    ######## Cost ########
    xref = ca.vertcat(*[param_manager.get(f"xref{i}") for i in range(4)])
    uref = param_manager.get("uref")
    yref = ca.vertcat(xref, uref)  # type:ignore
    yref_e = yref[: ocp.dims.nx]
    y = ca.vertcat(ocp.model.x, ocp.model.u)
    y_e = ocp.model.x

    q_diag_sqrt = param_manager.get("q_diag_sqrt")
    r_diag_sqrt = param_manager.get("r_diag_sqrt")
    W_sqrt = ca.diag(ca.vertcat(q_diag_sqrt, r_diag_sqrt))
    W = W_sqrt @ W_sqrt.T
    W_e = W[: ocp.dims.nx, : ocp.dims.nx]

    if cost_type == "EXTERNAL":
        ocp.cost.cost_type = ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = 0.5 * (y - yref).T @ W @ (y - yref)
        ocp.model.cost_expr_ext_cost_e = 0.5 * (y_e - yref_e).T @ W_e @ (y_e - yref_e)
        ocp.solver_options.hessian_approx = "EXACT"
    elif cost_type == "NONLINEAR_LS":
        ocp.cost.cost_type = ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = W
        ocp.cost.yref = yref
        ocp.model.cost_y_expr = y
        ocp.cost.W_e = W_e
        ocp.cost.yref_e = yref_e
        ocp.model.cost_y_expr_e = y_e
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    else:
        raise ValueError(f"Cost type {cost_type} not supported.")

    ######## Constraints ########
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0])

    if use_force_constraint:
        ocp.constraints.lbu = np.array([-Fmax])
        ocp.constraints.ubu = np.array([+Fmax])
        ocp.constraints.idxbu = np.array([0])

    if use_position_constraint:
        ocp.constraints.lbx = np.array([-x_threshold])
        ocp.constraints.ubx = -ocp.constraints.lbx
        ocp.constraints.idxbx = np.array([0])
        ocp.constraints.lbx_e = np.array([-x_threshold])
        ocp.constraints.ubx_e = -ocp.constraints.lbx_e
        ocp.constraints.idxbx_e = np.array([0])

    ######## Solver configuration ########
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 10
    ocp.solver_options.reg_epsilon = 5e-2
    ocp.solver_options.levenberg_marquardt = 1e-6

    return ocp
