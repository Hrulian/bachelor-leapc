import casadi as ca
import numpy as np
from acados_template import AcadosOcp
from pendulum_model import export_pendulum_model

from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager


def setup_ocp_with_manager(Fmax: float, N_horizon: int, Tf: float, 
              param_manager: AcadosParameterManager) -> AcadosOcp:
    """Create an ocp object to formulate the OCP for the cartpole system.
    
    Args:
        Fmax: Maximum force that can be applied to the cart.
        N_horizon: Number of steps in the prediction horizon.
        Tf: Total time horizon of the prediction horizon.
        param_manager: Parameter manager that holds the parameters to be used in the ocp.
    """
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()
    
    # set model
    ocp.model = export_pendulum_model(dt=Tf/N_horizon)
    ocp.solver_options.integrator_type = "DISCRETE"

    nx = ocp.model.x.rows()
    
    # TODO Exercise 2.3: assign the parameters to the ocp using the param_manager
    param_manager.assign_to_ocp(ocp)

    # ======================= set cost =======================
    # NOTE: NONLINEAR_LS pseudo: 0.5 * (y_expr - yref)^T W (y_expr - yref)
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    # weight matrices
    q_diag_sqrt = param_manager.get("q_diag_sqrt")
    r_diag_sqrt = param_manager.get("r_diag_sqrt")
    W_sqrt = ca.diag(ca.vertcat(q_diag_sqrt, r_diag_sqrt))
    ocp.cost.W = W_sqrt @ W_sqrt.T
    ocp.cost.W_e = ocp.cost.W[: nx, : nx]

    # expressions for ys
    ocp.model.cost_y_expr = ca.vertcat(ocp.model.x, ocp.model.u)
    ocp.model.cost_y_expr_e = ocp.model.x

    # references
    # TODO Exercise 2.3: Obtain the new AcadosParameters from the param_manager 
    # and use them to set the references below
    xref = ca.vertcat(*[param_manager.get(f"xref{i}") for i in range(1, 5)])
    uref = param_manager.get("uref")
    yref = ca.vertcat(xref, uref)
    yref_e = yref[: nx]
    ocp.cost.yref = yref
    ocp.cost.yref_e = yref_e

    # ============== set constraints ================
    # Actuator constraints
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


def create_cartpole_params(
) -> list[AcadosParameter]:
    """Returns a list of parameters used in the cartpole controller.
    """
    # TODO Exercise 2.2: Define new AcadosParameters, one for each reference in state and control 
    # (so, 5 in total). 
    # Make only the reference of the angle have the "learnable" interface, 
    # all other parameters should have the "fix" interface.
    return [
        # --------- weight matrix ---------
        AcadosParameter(
            "q_diag_sqrt", 
            default=np.sqrt(np.array([1e3, 1e3, 1e-2, 1e-2])), interface="fix"
        ),  # cost weights of state residuals
        AcadosParameter(
            "r_diag_sqrt", 
            default=np.sqrt(np.array([1e-2])), interface="fix"
        ),  # cost weights of control input residuals
        AcadosParameter(
            "xref1", default=np.array([0.0]), interface="fix"
        ),
        AcadosParameter(
            "xref2", default=np.array([0.0]), interface="learnable"
        ),
        AcadosParameter(
            "xref3", default=np.array([0.0]), interface="fix"
        ),
        AcadosParameter(
            "xref4", default=np.array([0.0]), interface="fix"
        ),
        AcadosParameter(
            "uref", default=np.array([0.0]), interface="fix"
        ),
    ]