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
    """Plot a cartpole CSV log with columns: t,x_m,theta,v_m_s,thetadot,u_force,u_pwm.

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
    u_force = []
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
                u_force.append(g('u_force'))
                u_pwm.append(g('u_pwm'))
    except Exception as e:
        print('Could not read cartpole log:', e)
        return

    if any(t is None for t in times):
        times = list(range(len(x_m)))

    fig, axes = plt.subplots(6, 1, sharex=True, figsize=(10, 10))
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

    axes[4].plot(times, u_force, '-m')
    axes[4].set_ylabel('u_force (N)')
    axes[4].grid(True)

    axes[5].plot(times, u_pwm, '-k')
    axes[5].set_ylabel('u (PWM)')
    axes[5].set_xlabel('time (s)')
    axes[5].grid(True)

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


def plot_policy_heatmap(
    actor_path: str,
    v_fixed: float = 0.0,
    thetadot_fixed: float = 0.0,
    x_range: tuple[float, float] = (-0.3, 0.3),
    theta_range: tuple[float, float] = (-np.pi, np.pi),
    resolution: int = 50,
    plt_show: bool = True,
    save_path: str | None = None
):
    """Create a heatmap showing theta_ref output from policy for different (x, theta) states.
    
    Args:
        actor_path: Path to actor.pth checkpoint file
        v_fixed: Fixed cart velocity value (m/s)
        thetadot_fixed: Fixed pole angular velocity (rad/s)
        x_range: Range of cart positions to plot (min, max) in meters
        theta_range: Range of pole angles to plot (min, max) in radians
        resolution: Number of grid points along each axis
        plt_show: Whether to display the plot
        save_path: Optional path to save the figure
    """
    try:
        import torch
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError as e:
        print(f"Required libraries not available: {e}")
        return
    
    # Import required modules for loading the actor
    try:
        from bachelor.acados_cartpole.real_sac_zop.my_sac_zop import MpcSacActor, SacZopTrainerConfig
        from bachelor.acados_cartpole.my_planner import CartPolePlannerConfig, CartPolePlanner, create_custom_cartpole_params
        from leap_c.planner import ControllerFromPlanner
        from leap_c.torch.nn.extractor import get_extractor_cls
        import gymnasium as gym
    except ImportError as e:
        print(f"Failed to import cartpole modules: {e}")
        return
    
    # Setup planner and actor
    device = "cpu"
    cfg_planner = CartPolePlannerConfig()
    params = create_custom_cartpole_params("global", cfg_planner.N_horizon)
    planner = CartPolePlanner(cfg_planner, params)
    controller_wrapped = ControllerFromPlanner(planner)
    
    # Define observation space
    _x_thr = getattr(cfg_planner, "x_threshold", 0.39)
    obs_low = np.array([-_x_thr, -np.pi, -5, -21], dtype=np.float32)
    obs_high = np.array([_x_thr, np.pi, 5, 21], dtype=np.float32)
    obs_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
    action_space = controller_wrapped.param_space
    
    # SacZop config
    cfg_saczop = SacZopTrainerConfig()
    
    # Initialize actor (same as in real_sac_zop.py)
    extractor_cls = get_extractor_cls("identity")
    actor = MpcSacActor(
        extractor_cls=extractor_cls,
        observation_space=obs_space,
        controller=controller_wrapped,
        distribution_name=cfg_saczop.distribution_name,
        mlp_cfg=cfg_saczop.actor_mlp,
        init_param_with_default=cfg_saczop.init_param_with_default,
    ).to(device)
    
    # Load actor weights
    try:
        actor.load_state_dict(torch.load(actor_path, map_location=device))
        actor.eval()
        print(f"Loaded actor from {actor_path}")
    except Exception as e:
        print(f"Failed to load actor: {e}")
        return
    
    # Create meshgrid
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    theta_vals = np.linspace(theta_range[0], theta_range[1], resolution)
    X, Theta = np.meshgrid(x_vals, theta_vals)
    
    # Initialize output arrays for theta_ref and action values
    theta_ref_grid = np.zeros_like(X)
    action_grid = np.zeros_like(X)
    
    # Evaluate policy for each (x, theta) combination
    print("Evaluating policy on grid...")
    with torch.no_grad():
        for i in range(resolution):
            for j in range(resolution):
                # Create state: [x, theta, v, thetadot]
                state = np.array([X[i, j], Theta[i, j], v_fixed, thetadot_fixed], dtype=np.float32)
                obs = torch.from_numpy(state).float().unsqueeze(0).to(device)
                
                # Get policy output with action (full forward pass)
                pi_out = actor(obs, ctx=None, deterministic=True, only_param=False)
                param = pi_out.param[0].cpu().numpy()
                action = pi_out.action[0].cpu().numpy()
                
                # Debug: print first param to see structure
                if i == 0 and j == 0:
                    print(f"First param shape: {param.shape}, size: {param.size}")
                    print(f"First param values: {param}")
                    print(f"First action shape: {action.shape}, values: {action}")
                
                # Extract theta_ref (param[0] is theta_ref for global interface)
                # The policy outputs only learnable parameters (just theta_ref in this case)
                if param.size > 0:
                    theta_ref_grid[i, j] = param[0]
                else:
                    theta_ref_grid[i, j] = 0.0
                
                # Extract action (force)
                action_grid[i, j] = action[0]
    
    # Debug output
    print(f"theta_ref_grid min: {theta_ref_grid.min():.4f}, max: {theta_ref_grid.max():.4f}, mean: {theta_ref_grid.mean():.4f}")
    print(f"Number of unique values: {len(np.unique(theta_ref_grid))}")
    print(f"action_grid min: {action_grid.min():.4f}, max: {action_grid.max():.4f}, mean: {action_grid.mean():.4f}")
    
    # ========== First figure: theta_ref ==========
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    vmin_theta = theta_ref_grid.min()
    vmax_theta = theta_ref_grid.max()

    if vmin_theta < 0.0 < vmax_theta:
        norm_theta = TwoSlopeNorm(vmin=vmin_theta, vcenter=0.0, vmax=vmax_theta)
        cmap_theta = 'RdYlBu_r'
    else:
        norm_theta = None
        cmap_theta = 'viridis'

    im1 = ax1.contourf(X, Theta, theta_ref_grid, levels=30, cmap=cmap_theta, norm=norm_theta)
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label(r'$\theta_{\mathrm{ref}}$ (rad)', rotation=270, labelpad=20)
    
    ax1.set_xlabel('Cart Position x (m)')
    ax1.set_ylabel('Pole Angle θ (rad)')
    ax1.set_title(f'Policy θ_ref Output (v={v_fixed:.2f} m/s, θ̇={thetadot_fixed:.2f} rad/s)')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.plot(0, 0, 'k*', markersize=10, label='Origin')
    ax1.legend()
    
    plt.tight_layout()
    
    # ========== Second figure: action (force) ==========
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    vmin_action = action_grid.min()
    vmax_action = action_grid.max()

    if vmin_action < 0.0 < vmax_action:
        norm_action = TwoSlopeNorm(vmin=vmin_action, vcenter=0.0, vmax=vmax_action)
        cmap_action = 'RdYlBu_r'
    else:
        norm_action = None
        cmap_action = 'viridis'

    im2 = ax2.contourf(X, Theta, action_grid, levels=30, cmap=cmap_action, norm=norm_action)
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label(r'$F$ (N)', rotation=270, labelpad=20)
    
    ax2.set_xlabel('Cart Position x (m)')
    ax2.set_ylabel('Pole Angle θ (rad)')
    ax2.set_title(f'Policy Force Output (v={v_fixed:.2f} m/s, θ̇={thetadot_fixed:.2f} rad/s)')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.plot(0, 0, 'k*', markersize=10, label='Origin')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        try:
            # Save both figures with different names
            save_path_base = Path(save_path).stem
            save_path_ext = Path(save_path).suffix
            save_path_dir = Path(save_path).parent
            
            save_path_theta = save_path_dir / f"{save_path_base}_theta_ref{save_path_ext}"
            save_path_force = save_path_dir / f"{save_path_base}_force{save_path_ext}"
            
            fig1.savefig(save_path_theta, bbox_inches='tight', dpi=150)
            fig2.savefig(save_path_force, bbox_inches='tight', dpi=150)
            print(f'Saved theta_ref heatmap to {save_path_theta}')
            print(f'Saved force heatmap to {save_path_force}')
        except Exception as e:
            print(f'Failed to save heatmaps: {e}')
    
    if plt_show:
        plt.show()



