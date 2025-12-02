# 

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from pathlib import Path
from acados_template import latexify_plot

def plot_pendulum(t, u_max, U, X_true, latexify=False, plt_show=True, time_label='$t$', x_labels=None, u_labels=None):
    """
    Params:
        t: time values of the discretization
        u_max: maximum absolute value of u
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X_true: arrray with shape (N_sim, nx)
        latexify: latex style plots
    """

    if latexify:
        latexify_plot()

    nx = X_true.shape[1]
    fig, axes = plt.subplots(nx+1, 1, sharex=True)

    for i in range(nx):
        axes[i].plot(t, X_true[:, i])
        axes[i].grid()
        if x_labels is not None:
            axes[i].set_ylabel(x_labels[i])
        else:
            axes[i].set_ylabel(f'$x_{i}$')

    axes[-1].step(t, np.append([U[0]], U))

    if u_labels is not None:
        axes[-1].set_ylabel(u_labels[0])
    else:
        axes[-1].set_ylabel('$u$')

    axes[-1].hlines(u_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
    axes[-1].hlines(-u_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
    axes[-1].set_ylim([-1.2*u_max, 1.2*u_max])
    axes[-1].set_xlim(t[0], t[-1])
    axes[-1].set_xlabel(time_label)
    axes[-1].grid()

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

    fig.align_ylabels()

    if plt_show:
        plt.show()


def plot_sol_and_sens(x_values:np.ndarray, pis: list[np.ndarray], senss: list[np.ndarray],
                      pi_labels: list[str], sens_labels: list[str], latexify: bool = True):
    """Plot solutions and their sensitivities."""

    if latexify:
        latexify_plot()

    # Create 2 subplots horizontally stacked
    _, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    # Now plot all pis with their labels in the first subplot
    for pi, label in zip(pis, pi_labels):
        axes[0].plot(x_values, pi, label=label)
        # Set the label on the x axis
    axes[0].set_title('Solutions')
    axes[0].set_xlabel(r'$\theta_{\mathrm{ref}}$')
    axes[0].set_ylabel(r'$u_0$')
    axes[0].legend()
    axes[0].grid()
    axes[0].set_xlim(min(x_values), max(x_values))

    # Now plot all sens with their labels in the second subplot
    for sens, label in zip(senss, sens_labels):
        axes[1].plot(x_values, sens, label=label)
    axes[1].set_title('Sensitivities')
    axes[1].set_xlabel(r'$\theta_{\mathrm{ref}}$')
    axes[1].set_ylabel(r'$\frac{\partial u_0}{\partial \theta_{\mathrm{ref}}}$')
    axes[1].set_ylim(bottom=-100)
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()

    # Save the figure
    plt.show()
    
    

def plot_cartpole_log(path: str, plt_show: bool = True, save_path: str | None = None):
    """Plot a cartpole CSV log with columns: t,x_m,theta,v_m_s,thetadot,u_pwm.

    - `path` can be absolute or relative to the script.
    - `save_path` if provided will be used to save the generated figure (PNG/PDF/etc).
    - If matplotlib is not available the function will print an informative message.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib required for plotting cartpole log:", e)
        return

    times = []
    x_m = []
    theta = []
    v_ms = []
    thetadot = []
    u_pwm = []

    try:
        with open(path, newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    times.append(float(row.get('t', '')))
                except Exception:
                    times.append(None)
                def g(name):
                    try:
                        return float(row.get(name, '') or 0.0)
                    except Exception:
                        return 0.0
                x_m.append(g('x_m'))
                theta.append(g('theta'))
                v_ms.append(g('v_m_s'))
                thetadot.append(g('thetadot'))
                u_pwm.append(g('u_pwm'))
    except Exception as e:
        print('Could not read cartpole log:', e)
        return

    if any(t is None for t in times):
        times = list(range(len(x_m)))

    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 8))
    axes[0].plot(times, x_m, '-b')
    axes[0].set_ylabel('x (m)')
    axes[0].grid(True)

    axes[1].plot(times, theta, '-r')
    axes[1].set_ylabel('theta (rad)')
    axes[1].grid(True)

    axes[2].plot(times, v_ms, '-g')
    axes[2].set_ylabel('v (m/s)')
    axes[2].grid(True)

    axes[3].plot(times, thetadot, '-c')
    axes[3].set_ylabel('thetadot (rad/s)')
    axes[3].grid(True)

    axes[4].plot(times, u_pwm, '-k')
    axes[4].set_ylabel('u (PWM)')
    axes[4].set_xlabel('time (s)')
    axes[4].grid(True)

    fig.suptitle('Cartpole states and control over time')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Determine save target: use explicit `save_path` if provided,
    # otherwise save into the current working directory using the
    # CSV filename stem (so the plot is always saved next to where
    # you invoked the script).
    try:
        if save_path is None:
            default_name = Path(path).stem + '.png'
            save_target = os.path.join(os.getcwd(), default_name)
        else:
            save_target = save_path

        try:
            fig.savefig(save_target, bbox_inches='tight')
            print('Saved plot to', save_target)
        except Exception as e:
            print('Failed to save plot to', save_target, ':', e)
    except Exception as e:
        print('Unexpected error while saving plot:', e)

    if plt_show:
        plt.show()



