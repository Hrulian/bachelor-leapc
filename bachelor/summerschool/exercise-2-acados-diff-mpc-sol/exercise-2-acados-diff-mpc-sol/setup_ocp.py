import casadi as ca
import numpy as np
from acados_template import AcadosOcp
from pendulum_model import export_pendulum_model


def setup_ocp(Fmax: float, N_horizon: int, Tf: float) -> AcadosOcp:
    """Create an ocp object to formulate the OCP for the cartpole system.
    
    Args:
        Fmax: Maximum force that can be applied to the cart.
        N_horizon: Number of steps in the prediction horizon.
        Tf: Total time horizon of the prediction horizon.
    """
    # TODO Exercise 1.2: Fill in the blanks in this method
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()
    
    # set model
    # Contains the dynamics of the CartPole environment
    ocp.model = export_pendulum_model(dt=Tf/N_horizon)
    ocp.solver_options.integrator_type = "DISCRETE"

    nx = ocp.model.x.rows()

    # ======================= set cost =======================
    # NOTE: NONLINEAR_LS pseudo: 0.5 * (y_expr - yref)^T W (y_expr - yref)
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    # weight matrices
    ocp.cost.W = np.diag([1e3, 1e3, 1e-2, 1e-2, 1e-2])  # state and input weights
    ocp.cost.W_e = ocp.cost.W[:nx, :nx]

    # expressions for ys
    x = ocp.model.x
    u = ocp.model.u
    # NOTE (use x and u from the two lines above here)
    ocp.model.cost_y_expr = ca.vertcat(x, u)
    ocp.model.cost_y_expr_e = x

    # references
    ocp.cost.yref = np.zeros((nx + 1,))  # desired state and input
    ocp.cost.yref_e = ocp.cost.yref[:nx]

    # ============== set constraints ================
    # Actuator constraints
    # NOTE: Use Fmax
    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])

    # Will be changed in the closed loop
    ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])

    # ============== set ocp options ================
    # prediction horizon
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = Tf

    # additional ocp options
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.qp_tol = 1e-8
    
    # ============= Finalize =============
    ocp.code_export_directory = 'c_generated_code_ocp'

    return ocp