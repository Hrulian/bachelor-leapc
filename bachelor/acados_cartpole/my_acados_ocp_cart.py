# Simplified cart-only model: only the cart is modelled, a force acts on it,
# the pendulum is completely removed.
#
#   state x = [pos, vel]      (nx = 2)
#   input u = [F]             (nu = 1)
#   dynamics: pos_dot = vel
#             vel_dot = (F - F_fric) / M
#
# Mirrors the interface of my_acados_ocp.py (same function names so it can be
# swapped in by my_planner.py), but only the cart position reference is learnable.

from typing import Literal

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosModel, AcadosOcp
from leap_c.examples.utils.casadi import integrate_erk4
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

CartPoleAcadosParamInterface = Literal["global", "stagewise"]
"""Determines the exposed parameter interface of the controller.
"global" means that learnable parameters are the same for all stages of the horizon,
while "stagewise" means that learnable parameters can vary between stages.
"""
CartPoleAcadosCostType = Literal["EXTERNAL", "NONLINEAR_LS"]
"""The type of cost to use, either "EXTERNAL" or "NONLINEAR_LS". Both model the same cost function,
but the former uses an exact Hessian in the optimization, while the latter uses a
Gauss-Newton Hessian approximation.
"""

# index of the cart states [pos, vel] inside the full RL observation
# [x, theta, v, thetadot]. Used by the planner to feed the right x0 to this OCP.
CART_OBS_IDX = [0, 2]


def create_custom_cartpole_params(
    param_interface: CartPoleAcadosParamInterface,
    N_horizon: int = 50,
) -> list[AcadosParameter]:
    """Returns the parameters used by the simplified cart-only controller.

    Only the cart position reference (``xref0``) is learnable; everything else is
    fixed.

    Args:
        param_interface: Determines the exposed parameter interface of the controller.
        N_horizon: The number of steps in the MPC horizon.
    """
    return [
        # Dynamics parameters
        AcadosParameter("M", default=np.array([0.17444])),  # mass of the cart [kg]
        # Cost matrix factorization parameters
        AcadosParameter(
            "q_diag_sqrt", default=np.sqrt(np.array([1e3, 1.0]))
        ),  # cost weights of state residuals [pos, vel]
        AcadosParameter(
            "r_diag_sqrt", default=np.sqrt(np.array([1e-1]))
        ),  # cost weights of control input residuals [F]
        # Reference parameters
        AcadosParameter(
            "xref0",
            default=np.array([0.0]),
            space=gym.spaces.Box(
                low=np.array([-0.4]),
                high=np.array([0.4]),
                dtype=np.float64,
            ),
            interface="learnable",
            end_stages=list(range(N_horizon + 1)) if param_interface == "stagewise" else [],
        ),  # reference cart position (learnable)
        AcadosParameter(
            "xref1",
            default=np.array([0.0]),
            interface="non-learnable",
        ),  # reference cart velocity
        AcadosParameter(
            "uref",
            default=np.array([0.0]),
            interface="non-learnable",
        ),  # reference u
    ]


def define_f_expl_expr(model: AcadosModel, param_manager: AcadosParameterManager) -> ca.SX:
    M = param_manager.get("M")

    v = model.x[1]
    F = model.u[0]

    # friction parameters found experimentally (kept disabled to mirror my_acados_ocp.py)
    B_V = 8.450926
    F_C = 1.32038
    F_fric = 0  # B_V * v + F_C * differentiable_signum(v)

    F_eff = F - F_fric

    f_expl = ca.vertcat(
        v,
        F_eff / M,
    )

    return f_expl  # type: ignore


def export_parametric_ocp(
    param_manager: AcadosParameterManager,
    cost_type: CartPoleAcadosCostType = "NONLINEAR_LS",
    name: str = "cartpole_cart",
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

    ocp.dims.nx = 2
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
    xref = ca.vertcat(*[param_manager.get(f"xref{i}") for i in range(2)])
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
        ocp.cost.cost_type = cost_type
        ocp.model.cost_expr_ext_cost = 0.5 * (y - yref).T @ W @ (y - yref)

        ocp.cost.cost_type_e = cost_type
        ocp.model.cost_expr_ext_cost_e = 0.5 * (y_e - yref_e).T @ W_e @ (y_e - yref_e)

        ocp.solver_options.hessian_approx = "EXACT"
    elif cost_type == "NONLINEAR_LS":
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        ocp.cost.W = W
        ocp.cost.yref = yref
        ocp.model.cost_y_expr = y

        ocp.cost.W_e = W_e
        ocp.cost.yref_e = yref_e
        ocp.model.cost_y_expr_e = y_e

        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    else:
        raise ValueError(f"Cost type {cost_type} not supported. Use 'EXTERNAL' or 'NONLINEAR_LS'.")

    ######## Constraints ########
    ocp.constraints.idxbx_0 = np.array([0, 1])
    ocp.constraints.x0 = np.array([0.0, 0.0])  # [pos, vel]

    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])

    # Position constraints on the cart temporarily disabled for testing.
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

    # max iterations for the QP/NLP solver so if SQP fails we stop early
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 10

    ocp.solver_options.reg_epsilon = 5e-2
    ocp.solver_options.levenberg_marquardt = 1e-6

    return ocp
