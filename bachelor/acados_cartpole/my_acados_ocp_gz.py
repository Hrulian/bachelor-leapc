# Full cart+pendulum model, set up the way Gros & Zanon actually intend it:
#
#   * theta   = the LEARNED cost parametrization. Here the pole-angle reference
#               xref1, exactly like my_acados_ocp.py ("full"). RL tunes it.
#   * d       = the EXPLORATION channel. A gradient disturbance d^T u_0 added to
#               the stage-0 cost, DRAWN from a distribution each step, never
#               learned. See Gros & Zanon, "Towards Safe Reinforcement Learning
#               Using NMPC and Policy Gradients: Part I - Stochastic case"
#               (arXiv:1906.04057), Sec. IV-C, Eq. (43)+(45):
#
#                   u^d(s, theta, d) = arg min_u  Phi(x, u, theta) + d^T u_0
#                                      s.t.       f(...) = 0,  h(...) <= 0
#
#               "the parameter d is drawn from an arbitrary probability
#                distribution [...] can, e.g., be a simple Gaussian" and it
#               "introduces stochasticity in the inputs a generated".
#
# This is the inverse of my_acados_ocp_du0.py, which learns d and freezes the
# references. Here d is noise and the reference is learned — the split the paper
# prescribes. The key property is unchanged: because the perturbed action still
# falls out of the constrained NLP, every realization satisfies the input and
# state constraints by construction (paper: "any realization [...] is in S(s) by
# construction"), unlike additive noise on the action.
#
#   state x = [pos, theta, vel, thetadot]   (nx = 4, full model)
#   input u = [F]                           (nu = 1)
#   param vector handed to the solver = [xref1, d_u0]  (in THAT order)
#
# Both entries sit in the "learnable" interface because that is simply leap_c's
# per-call-settable parameter channel; only xref1 is actually optimized by RL,
# d_u0 is filled with the sampled noise each step (and with 0 for the greedy /
# evaluation action).
#
# Why the perturbation is implemented as a shifted stage-0 input reference:
#
#       0.5*r*(u - uref)^2 + d*u = 0.5*r*(u - (uref - d/r))^2 + const
#
# so for NONLINEAR_LS we use yref_0 = [xref, uref - d/r] on the initial shooting
# node only; stages 1..N-1 and the terminal node keep the unperturbed reference.
# The additive constant does not depend on u, so the argmin — and therefore the
# policy — is unchanged. For "EXTERNAL" the term d*u_0 is added literally.
# d = 0 reproduces the unperturbed MPC exactly.
#
# NOTE on the size of d: the applied force is NOT uref_0; it follows from the
# reduced Hessian of the whole OCP, which the state weights dominate
# (q = [1e3, 1e3, 1, 1] vs. r = 0.1). Measured du_0/dd ~ -0.19 upright and
# ~ -3e-4 hanging, so a useful exploration std for d is on the order of 1e1-1e2,
# NOT r*Fmax = 2. See GZPlannerConfig.explore_sigma_d.

from typing import Literal

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosModel, AcadosOcp
from leap_c.examples.utils.casadi import integrate_erk4
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

CartPoleAcadosParamInterface = Literal["global", "stagewise"]
"""Determines the exposed parameter interface of the controller.

NOTE: d_u0 only enters the stage-0 cost, so "stagewise" would add parameters
that have no effect on the solution. Keep "global" here.
"""
CartPoleAcadosCostType = Literal["EXTERNAL", "NONLINEAR_LS"]
"""Both model the same cost; EXTERNAL uses an exact Hessian, NONLINEAR_LS a
Gauss-Newton approximation."""

# Input cost weight r (as its square root, the form the OCP is parametrized in).
R_DIAG = np.array([1e-1])


def create_custom_cartpole_params(
    param_interface: CartPoleAcadosParamInterface,
    N_horizon: int = 50,
    d_max: float = 300.0,
) -> list[AcadosParameter]:
    """Parameters of the Gros & Zanon-style cartpole controller.

    LEARNED: ``xref1`` (pole-angle reference), identical to my_acados_ocp.py.
    SAMPLED (not learned): ``d_u0``, the gradient disturbance used for exploration.

    Args:
        param_interface: Determines the exposed parameter interface.
        N_horizon: The number of steps in the MPC horizon.
        d_max: Half-width of the ``d_u0`` box. This is NOT a tuning knob for
            authority (that is what the reference is for) — it only has to be
            wide enough that the sampled noise is not clipped, so keep it a few
            sigma above ``explore_sigma_d``.
    """
    return [
        # Dynamics parameters (identical to my_acados_ocp.py)
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
        # References. Only the pole angle is learnable — same as "full".
        AcadosParameter(
            "xref0",
            default=np.array([0.0]),
            interface="non-learnable",
        ),  # reference position
        AcadosParameter(
            "xref1",
            default=np.array([0.0]),
            space=gym.spaces.Box(
                low=np.array([-np.pi]),
                high=np.array([np.pi]),
                dtype=np.float64,
            ),
            interface="learnable",
            end_stages=list(range(N_horizon + 1)) if param_interface == "stagewise" else [],
        ),  # reference theta  <-- THE LEARNED PARAMETER (theta)
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
        # Gradient disturbance, Gros & Zanon (arXiv:1906.04057) Eq. (45).
        # Settable per call, but filled with SAMPLED NOISE, never optimized.
        AcadosParameter(
            "d_u0",
            default=np.array([0.0]),  # 0 => unperturbed MPC (greedy action)
            space=gym.spaces.Box(
                low=np.array([-abs(float(d_max))]),
                high=np.array([+abs(float(d_max))]),
                dtype=np.float64,
            ),
            interface="learnable",
            end_stages=[],
        ),  # <-- THE EXPLORATION CHANNEL (d)
    ]


def define_f_expl_expr(model: AcadosModel, param_manager: AcadosParameterManager) -> ca.SX:
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
    name: str = "cartpole_gz",
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

    # ---- Exploration: gradient disturbance, Eq. (45) ------------------------
    # Phi_d = Phi + d^T u_0, folded into a shifted stage-0 input reference.
    d_u0 = param_manager.get("d_u0")
    r = r_diag_sqrt**2  # scalar, nu = 1
    uref_0 = uref - d_u0 / r
    yref_0 = ca.vertcat(xref, uref_0)  # type:ignore
    # ------------------------------------------------------------------------

    if cost_type == "EXTERNAL":
        ocp.cost.cost_type_0 = cost_type
        ocp.model.cost_expr_ext_cost_0 = (
            0.5 * (y - yref).T @ W @ (y - yref) + d_u0 * ocp.model.u[0]
        )

        ocp.cost.cost_type = cost_type
        ocp.model.cost_expr_ext_cost = 0.5 * (y - yref).T @ W @ (y - yref)

        ocp.cost.cost_type_e = cost_type
        ocp.model.cost_expr_ext_cost_e = 0.5 * (y_e - yref_e).T @ W_e @ (y_e - yref_e)

        ocp.solver_options.hessian_approx = "EXACT"
    elif cost_type == "NONLINEAR_LS":
        ocp.cost.cost_type_0 = "NONLINEAR_LS"
        ocp.cost.W_0 = W
        ocp.cost.yref_0 = yref_0
        ocp.model.cost_y_expr_0 = y

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.W = W
        ocp.cost.yref = yref
        ocp.model.cost_y_expr = y

        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W_e = W_e
        ocp.cost.yref_e = yref_e
        ocp.model.cost_y_expr_e = y_e

        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    else:
        raise ValueError(f"Cost type {cost_type} not supported. Use 'EXTERNAL' or 'NONLINEAR_LS'.")

    ######## Constraints ########
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.array([0.0, 0, 0.0, 0.0])

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

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 10

    ocp.solver_options.reg_epsilon = 5e-2
    ocp.solver_options.levenberg_marquardt = 1e-6

    return ocp
