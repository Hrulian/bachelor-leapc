import numpy as np

from leap_c.ocp.acados.parameters import AcadosParameter


def create_custom_cartpole_params(
) -> list[AcadosParameter]:
    """Returns a list of parameters used in the cartpole controller.

    Args:
        N_horizon: The number of steps in the MPC horizon.
    """
    return [
        # Dynamics parameters
        AcadosParameter("M", default=np.array([0.5])),  # mass of the cart [kg]
        AcadosParameter("m", default=np.array([0.06])),  # mass of the ball [kg]
        AcadosParameter("g", default=np.array([9.81])),  # gravity constant [m/s^2]
        AcadosParameter("l", default=np.array([0.8])),  # length of the rod [m]
        # Cost matrix factorization parameters
        AcadosParameter(
            "q_diag_sqrt", default=np.sqrt(np.array([2e3, 2e3, 1e-2, 1e-2]))
        ),  # cost weights of state residuals
        AcadosParameter(
            "r_diag_sqrt", default=np.sqrt(np.array([2e-1]))
        ),  # cost weights of control input residuals
        # Reference parameters
        AcadosParameter(
            "xref0",
            default=np.array([0.0]),
            interface="non-learnable",
        ),  # reference position
        # TODO: glaube hier muss ich irgendwie den space definieren. 
        # muss mein Winkel kontinuirlich sein? oder einfach nur [-pi,pi]
        # mit sprung unten ODER smoothen?
        AcadosParameter(
            "xref1",
            default=np.array([0.0]),
            interface="learnable",
        ),  # reference theta
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
    ]
