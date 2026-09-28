# Full cart+pendulum model (identical dynamics to my_acados_ocp.py), BUT nothing
# in the reference is learnable any more. Instead the RL agent learns a LINEAR
# COST PERTURBATION on the first input, following Gros & Zanon (2021), Eq. (11):
#
#       Phi_d(x, u) = Phi(x, u) + d^T u_0
#
#   state x = [pos, theta, vel, thetadot]   (nx = 4, full model)
#   input u = [F]                           (nu = 1)
#   learnable = d_u0  (gradient perturbation on u_0)
#   every reference (xref0..3, uref) is fixed at 0 and non-learnable
#
# Why this is equivalent to a shifted input reference on stage 0:
#
#       0.5*r*(u - uref)^2 + d*u = 0.5*r*(u - (uref - d/r))^2 + const
#
# so for a NONLINEAR_LS cost the perturbation is implemented exactly by using
# yref_0 = [xref, uref - d/r] on the initial shooting node only, while stages
# 1..N-1 and the terminal node keep the unperturbed reference. The constant
# d^2/(2r) that the completion of the square adds does not depend on u, so the
# argmin — and therefore the policy and its sensitivity dpi/dd — is unchanged.
# (The reported cost VALUE differs from Phi + d*u_0 by that constant; only
# relevant if you ever read out the optimal value function itself.)
#
# For cost_type "EXTERNAL" the term d*u_0 is added literally to the stage-0 cost
# instead, so both cost types describe the same perturbed OCP.
#
# Bounds on d: at the optimum the perturbation can at most drag u_0 across the
# whole admissible force range, i.e. |d| <= r * Fmax. With r = 0.1 and
# Fmax = 20 N that is |d| <= 2.0. d = 0 reproduces the baseline MPC exactly.
#
# Everything else (dynamics, constraints, solver options) is byte-for-byte the
# same as my_acados_ocp.py; the function names mirror it so the file can be
# bound by my_planner_du0.py exactly like my_planner.py binds my_acados_ocp.py.

from typing import Literal

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosModel, AcadosOcp
from leap_c import ocp
from leap_c.examples.utils.casadi import integrate_erk4
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from bachelor.acados_cartpole.my_helpers import differentiable_signum

CartPoleAcadosParamInterface = Literal["global", "stagewise"]
"""Determines the exposed parameter interface of the controller.
"global" means that learnable parameters are the same for all stages of the horizon,
while "stagewise" means that learnable parameters can vary between stages.

NOTE: d_u0 only enters the stage-0 cost, so "stagewise" adds parameters that have
no effect on the solution. Keep "global" here.
"""
CartPoleAcadosCostType = Literal["EXTERNAL", "NONLINEAR_LS"]
"""The type of cost to use, either "EXTERNAL" or "NONLINEAR_LS". Both model the same cost function,
but the former uses an exact Hessian in the optimization, while the latter uses a
Gauss-Newton Hessian approximation.
"""

# Input cost weight r (as its square root, the form the OCP is parametrized in).
# Kept as a module constant because the bound on d_u0 is derived from it.
R_DIAG = np.array([1e-1])


def d_u0_bound(Fmax: float, r: np.ndarray = R_DIAG) -> np.ndarray:
    """Textbook bound on the cost perturbation, |d| <= r * Fmax.

    Rationale: -d/r is the shift of the stage-0 input reference, so this box is
    exactly the one that sweeps uref_0 across the whole admissible force range.

    WARNING — measured, not theoretical: this bound leaves the actor almost no
    authority over the force that is actually applied. u_0 is not determined by
    uref_0 alone but by the reduced Hessian of the whole OCP, which here is
    dominated by the state weights (q = [1e3, 1e3, 1, 1] vs. r = 0.1). Measured
    du_0/dd is between -0.19 (upright) and ~-3e-4 (hanging), so sweeping d over
    the full +-2.0 box moves the applied force by only ~0.001 - 1.4 N out of
    +-20 N. Spanning the force range needs |d| on the order of 1e2 - 1e3.
    Use `GradPertPlannerConfig.d_max` to override this box.
    """
    return np.asarray(r, dtype=np.float64) * float(Fmax)


def create_custom_cartpole_params(
    param_interface: CartPoleAcadosParamInterface,
    N_horizon: int = 50,
    Fmax: float = 20.0,
    d_max: float | None = None,
) -> list[AcadosParameter]:
    """Returns a list of parameters used in the cartpole controller.

    Full model, all references FIXED (non-learnable). The single LEARNABLE
    parameter is ``d_u0``, the linear cost perturbation on the first input
    (Gros & Zanon 2021, Eq. (11)).

    Args:
        param_interface: Determines the exposed parameter interface of the controller.
        N_horizon: The number of steps in the MPC horizon.
        Fmax: The force bound of the OCP; sets the box of ``d_u0`` via r * Fmax
            when `d_max` is None. Must match the Fmax handed to
            `export_parametric_ocp`.
        d_max: Explicit half-width of the ``d_u0`` box, overriding r * Fmax.
            See the warning in `d_u0_bound` — the textbook bound is far too
            small to give the actor real authority over the applied force.
    """
    d_box = d_u0_bound(Fmax) if d_max is None else np.array([abs(float(d_max))])
    return [
        # Dynamics parameters (full model — identical to my_acados_ocp.py)
        AcadosParameter("M", default=np.array([0.17444])),  # mass of the cart [kg]
        AcadosParameter("m", default=np.array([0.016])),  # mass of the rod [kg]
        AcadosParameter("g", default=np.array([9.81])),  # gravity constant [m/s^2]
        AcadosParameter("l", default=np.array([0.36])),  # length of the rod [m]
        AcadosParameter("Ip", default=np.array([0.0001728])),  # moment of inertia
        # Cost matrix factorization parameters
        AcadosParameter(
            "q_diag_sqrt", default=np.sqrt(np.array([1e3, 1e3, 1, 1]))
        ),  # cost weights of state residuals
        AcadosParameter(
            "r_diag_sqrt", default=np.sqrt(R_DIAG)
        ),  # cost weights of control input residuals
        # Reference parameters — ALL non-learnable here, that is the whole point:
        # the agent no longer moves a reference around, it perturbs the cost.
        AcadosParameter(
            "xref0",
            default=np.array([0.0]),
            interface="non-learnable",
        ),  # reference position
        AcadosParameter(
            "xref1",
            default=np.array([0.0]),
            interface="non-learnable",
        ),  # reference theta (fixed upright)
        AcadosParameter(
            "xref2",
            default=np.array([0.0]),
            interface="non-learnable",
        ),  # reference v
        AcadosParameter(
            "xref3",
            default=np.array([0.0]),
            interface="non-learnable",
        ),  # reference thetadot
        AcadosParameter(
            "uref",
            default=np.array([0.0]),
            interface="non-learnable",
        ),  # reference u
        # Linear cost perturbation on u_0 (Gros & Zanon 2021, Eq. (11)).
        # Cost: ... + d_u0 * u_0  <->  equivalent to uref_0 = uref - d_u0 / r.
        AcadosParameter(
            "d_u0",
            default=np.array([0.0]),  # 0 => exactly the baseline MPC
            space=gym.spaces.Box(
                low=-d_box,
                high=+d_box,
                dtype=np.float64,
            ),
            interface="learnable",
            end_stages=list(range(N_horizon + 1)) if param_interface == "stagewise" else [],
        ),
    ]


def define_f_expl_expr(model: AcadosModel, param_manager: AcadosParameterManager) -> ca.SX:
    M = param_manager.get("M")
    m = param_manager.get("m")
    g = param_manager.get("g")
    l = param_manager.get("l")
    Ip = param_manager.get("Ip")  # New: moment of inertia of the pendulum

    theta = model.x[1]
    v = model.x[2]
    dtheta = model.x[3]

    F = model.u[0]

    # modelling of friction parameters found experimentally
    B_V = 8.450926
    F_C = 1.32038

    #v_sign = differentiable_signum(v) # this version is differentiable
    F_fric = 0#B_V * v + F_C * v_sign

    F_eff = F - F_fric

    # pole friction (only viscous for now)
    d = 0.000081

    tau = 0#-d * dtheta

    # dynamics
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    denominator = M + m - m * cos_theta * cos_theta#(M + m) * (Ip + m * l**2) - (m * l * cos_theta)**2

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
    name: str = "cartpole_du0",
    Fmax: float = 10.0,
    x_threshold: float = 0.1,
    N_horizon: int = 20,
    T_horizon: float = 1.0,
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
    ocp.model.disc_dyn_expr = integrate_erk4(
        f_expl=f_expl,
        x=ocp.model.x,
        u=ocp.model.u,
        p=p,
        dt=dt,
    )

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

    # ---- Gradient perturbation, Gros & Zanon (2021) Eq. (11) ----------------
    # Phi_d = Phi + d^T u_0. With quadratic input cost
    #   0.5*r*(u - uref)^2 + d*u = 0.5*r*(u - (uref - d/r))^2 + const
    # => identical argmin, obtained by shifting the input reference on stage 0.
    d_u0 = param_manager.get("d_u0")
    r = r_diag_sqrt**2  # scalar, nu = 1
    uref_0 = uref - d_u0 / r
    yref_0 = ca.vertcat(xref, uref_0)  # type:ignore
    # ------------------------------------------------------------------------

    if cost_type == "EXTERNAL":
        # Stage 0: the perturbation term added literally.
        ocp.cost.cost_type_0 = cost_type
        ocp.model.cost_expr_ext_cost_0 = (
            0.5 * (y - yref).T @ W @ (y - yref) + d_u0 * ocp.model.u[0]
        )

        # Stages 1..N-1: unperturbed.
        ocp.cost.cost_type = cost_type
        ocp.model.cost_expr_ext_cost = 0.5 * (y - yref).T @ W @ (y - yref)

        ocp.cost.cost_type_e = cost_type
        ocp.model.cost_expr_ext_cost_e = 0.5 * (y_e - yref_e).T @ W_e @ (y_e - yref_e)

        ocp.solver_options.hessian_approx = "EXACT"
    elif cost_type == "NONLINEAR_LS":
        # Stage 0: perturbation folded into a shifted input reference.
        ocp.cost.cost_type_0 = "NONLINEAR_LS"
        ocp.cost.W_0 = W
        ocp.cost.yref_0 = yref_0
        ocp.model.cost_y_expr_0 = y

        # Stages 1..N-1: unchanged.
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.W = W
        ocp.cost.yref = yref
        ocp.model.cost_y_expr = y

        # Terminal.
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W_e = W_e
        ocp.cost.yref_e = yref_e
        ocp.model.cost_y_expr_e = y_e

        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    else:
        raise ValueError(f"Cost type {cost_type} not supported. Use 'EXTERNAL' or 'NONLINEAR_LS'.")

    ######## Constraints ########
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.array([0.0, 0, 0.0, 0.0]) # -> for now lets start in the upright position

    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])

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

    #addded max iterations for the QP solver and NLP solver so if SQP fails we stop early
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 10

    # those here i took from Katrin without really understanding them
    ocp.solver_options.reg_epsilon = 5e-2
    ocp.solver_options.levenberg_marquardt = 1e-6

    return ocp
