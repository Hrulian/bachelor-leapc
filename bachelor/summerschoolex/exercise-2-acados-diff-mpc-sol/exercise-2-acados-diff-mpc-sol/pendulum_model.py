import casadi as ca
from acados_template import AcadosModel


def integrate_erk4(f_expl: ca.SX, x: ca.SX, u: ca.SX, dt: float) -> ca.SX:
    """Integrate dynamics using the explicit RK4 method.

    Args:
        f_expl: The explicit dynamics function.
        x: The state vector.
        u: The control input vector.
        dt: The time step for integration.

    Returns:
        The updated state vector after integration.
    """
    ode = ca.Function("ode", [x, u], [f_expl])
    k1 = ode(x, u)
    k2 = ode(x + dt / 2 * k1, u)
    k3 = ode(x + dt / 2 * k2, u)
    k4 = ode(x + dt * k3, u)

    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def export_pendulum_model(dt: float) -> AcadosModel:
    model_name = 'pendulum_ode'

    # constants
    M = 1. # mass of the cart [kg] -> now estimated
    m = 0.1 # mass of the ball [kg]
    g = 9.81 # gravity constant [m/s^2]
    l = 0.8 # length of the rod [m]

    # set up states & controls
    x1      = ca.SX.sym('x1')
    theta   = ca.SX.sym('theta')
    v1      = ca.SX.sym('v1')
    dtheta  = ca.SX.sym('dtheta')

    x = ca.vertcat(x1, theta, v1, dtheta)

    F = ca.SX.sym('F')
    u = ca.vertcat(F)

    # xdot
    x1_dot      = ca.SX.sym('x1_dot')
    theta_dot   = ca.SX.sym('theta_dot')
    v1_dot      = ca.SX.sym('v1_dot')
    dtheta_dot  = ca.SX.sym('dtheta_dot')

    xdot = ca.vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot)

    # dynamics
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    denominator = M + m - m*cos_theta*cos_theta
    f_expl = ca.vertcat(v1,
                     dtheta,
                     (-m*l*sin_theta*dtheta*dtheta + m*g*cos_theta*sin_theta+F)/denominator,
                     (-m*l*cos_theta*sin_theta*dtheta*dtheta + F*cos_theta+(M+m)*g*sin_theta)/(l*denominator)
                     )

    model = AcadosModel()

    model.disc_dyn_expr = integrate_erk4(
        f_expl=f_expl,
        x=x,
        u=u,
        dt=dt,
    )
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    # store meta information
    model.x_labels = ['$x$ [m]', r'$\theta$ [rad]', '$v$ [m]', r'$\dot{\theta}$ [rad/s]']
    model.u_labels = ['$F$']
    model.t_label = '$t$ [s]'
    
    return model