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


def plot_eval_log(path: str, plt_show: bool = True, save_path: str | None = None):
    """Plot evaluation log with columns: t, x_m, theta_unwrapped, accumulated_reward.
    
    Creates a 3-subplot figure showing:
    - Cart position (x) over time
    - Pole angle (theta_unwrapped) over time
    - Accumulated reward over time
    
    Args:
        path: Path to the CSV log file
        plt_show: Whether to display the plot interactively
        save_path: Optional path to save the figure (PNG/PDF/etc)
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib required for plotting eval log:", e)
        return

    times = []
    x_m = []
    theta = []
    reward = []

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
                theta.append(g('theta_unwrapped'))
                reward.append(g('accumulated_reward'))
    except Exception as e:
        print('Could not read eval log:', e)
        return

    if any(t is None for t in times):
        times = list(range(len(x_m)))

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    
    axes[0].plot(times, x_m, '-b', linewidth=1.5)
    axes[0].set_ylabel('Cart Position x (m)', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)

    axes[1].plot(times, theta, '-r', linewidth=1.5)
    axes[1].set_ylabel('Pole Angle θ (rad)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    # Mark upright position (multiples of 2π)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8, label='upright')

    axes[2].plot(times, reward, '-g', linewidth=1.5)
    axes[2].set_ylabel('Accumulated Reward', fontsize=11)
    axes[2].set_xlabel('Time (s)', fontsize=11)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Controller Evaluation', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save figure
    try:
        if save_path is None:
            default_name = Path(path).stem + '.png'
            save_target = os.path.join(os.getcwd(), default_name)
        else:
            save_target = save_path

        try:
            fig.savefig(save_target, bbox_inches='tight', dpi=150)
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
    x_range: tuple[float, float] = (-0.39, 0.39),
    theta_range: tuple[float, float] = (-np.pi, np.pi),
    resolution: int = 50,
    plt_show: bool = True,
    save_path: str | None = None,
    episode_num: int | None = None
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
        from bachelor.acados_cartpole.real_sac_zop import MpcSacActor, SacZopTrainerConfig
        from bachelor.acados_cartpole.my_planner import CartPolePlannerConfig, CartPolePlanner, create_custom_cartpole_params
        from leap_c.planner import ControllerFromPlanner
        from leap_c.torch.nn.extractor import get_extractor_cls
        from leap_c.torch.rl.sac import SacCritic
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
    
    # Enable layer normalization for critic only (must match training settings)
    cfg_saczop.critic_mlp.norm_layer = "layer_norm"
    
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
    
    # Load critic
    critic_path = str(Path(actor_path).parent / "critic.pth")
    critic = SacCritic(
        extractor_cls=extractor_cls,
        observation_space=obs_space,
        action_space=action_space,
        mlp_cfg=cfg_saczop.critic_mlp,
        num_critics=cfg_saczop.num_critics,
    ).to(device)
    
    try:
        critic.load_state_dict(torch.load(critic_path, map_location=device))
        critic.eval()
        print(f"Loaded critic from {critic_path}")
    except Exception as e:
        print(f"Warning: Failed to load critic: {e}")
        critic = None
    
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
    
    # ========== Grid plots with 3x3 subplots ==========
    # Create a single large plot divided into 3x3 grid showing different (v, thetadot) combinations
    v_values = np.array([-1.0, 0.0, 1.0])  # 3 velocity values
    thetadot_values = np.array([-5.0, 0.0, 5.0])  # 3 angular velocity values
    
    print("Evaluating policy on grid with varying v and thetadot...")
    
    # Evaluate policy for all (v, thetadot) combinations
    # Create a 5D array: [v_idx, thetadot_idx, x_idx, theta_idx, output_type]
    # where output_type 0=theta_ref, 1=force
    grid_size_v = len(v_values)
    grid_size_thetadot = len(thetadot_values)
    
    theta_ref_full_grid = np.zeros((grid_size_v, grid_size_thetadot, resolution, resolution))
    action_full_grid = np.zeros((grid_size_v, grid_size_thetadot, resolution, resolution))
    critic_full_grid = np.zeros((grid_size_v, grid_size_thetadot, resolution, resolution))
    mpc_force_full_grid = np.zeros((grid_size_v, grid_size_thetadot, resolution, resolution))
    
    with torch.no_grad():
        for v_idx, v_val in enumerate(v_values):
            for td_idx, thetadot_val in enumerate(thetadot_values):
                for i in range(resolution):
                    for j in range(resolution):
                        state = np.array([X[i, j], Theta[i, j], v_val, thetadot_val], dtype=np.float32)
                        obs = torch.from_numpy(state).float().unsqueeze(0).to(device)
                        
                        pi_out = actor(obs, ctx=None, deterministic=True, only_param=False)
                        param = pi_out.param[0].cpu().numpy()
                        action = pi_out.action[0].cpu().numpy()
                        
                        if param.size > 0:
                            theta_ref_full_grid[v_idx, td_idx, i, j] = param[0]
                        else:
                            theta_ref_full_grid[v_idx, td_idx, i, j] = 0.0
                        
                        action_full_grid[v_idx, td_idx, i, j] = action[0]
                        
                        # Get critic Q-value if available
                        if critic is not None:
                            param_tensor = torch.from_numpy(param).float().unsqueeze(0).to(device)
                            q_values = critic(obs, param_tensor)  # Returns list of Q-values
                            q_value = torch.stack(q_values).min().cpu().numpy()
                            critic_full_grid[v_idx, td_idx, i, j] = q_value
                        
                        # Get MPC force output with default theta_ref=0
                        try:
                            # Call planner directly (it's an AcadosPlanner, needs tensor input)
                            # obs is already a tensor with batch dim from above: torch.Size([1, 4])
                            param_mpc = torch.tensor([[0.0]], dtype=torch.float32, device=device)  # theta_ref=0
                            _, u0, _, _, _ = planner(obs, param=param_mpc, ctx=None)
                            mpc_force = float(u0.detach().cpu().numpy().squeeze().item())
                            mpc_force_full_grid[v_idx, td_idx, i, j] = mpc_force
                        except Exception as e:
                            mpc_force_full_grid[v_idx, td_idx, i, j] = 0.0
    
    # Filter out inf and nan values to prevent overflow errors
    theta_ref_full_grid = np.nan_to_num(theta_ref_full_grid, nan=0.0, posinf=0.0, neginf=0.0)
    action_full_grid = np.nan_to_num(action_full_grid, nan=0.0, posinf=0.0, neginf=0.0)
    critic_full_grid = np.nan_to_num(critic_full_grid, nan=0.0, posinf=0.0, neginf=0.0)
    mpc_force_full_grid = np.nan_to_num(mpc_force_full_grid, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Debug output for force ranges
    print(f"\n=== Force Analysis ===")
    print(f"Policy force (with learned theta_ref):")
    print(f"  Range: [{action_full_grid.min():.4f}, {action_full_grid.max():.4f}]")
    print(f"  Mean: {action_full_grid.mean():.4f}, Std: {action_full_grid.std():.4f}")
    print(f"MPC force (with theta_ref=0):")
    print(f"  Range: [{mpc_force_full_grid.min():.4f}, {mpc_force_full_grid.max():.4f}]")
    print(f"  Mean: {mpc_force_full_grid.mean():.4f}, Std: {mpc_force_full_grid.std():.4f}")
    print(f"  Unique values: {len(np.unique(mpc_force_full_grid))}")
    
    # Additional clipping to prevent extreme values that might cause issues
    # Clip to reasonable ranges based on the physical system
    theta_ref_full_grid = np.clip(theta_ref_full_grid, -2*np.pi, 2*np.pi)
    action_full_grid = np.clip(action_full_grid, -100, 100)
    critic_full_grid = np.clip(critic_full_grid, -1000, 1000)
    mpc_force_full_grid = np.clip(mpc_force_full_grid, -100, 100)
    
    # Now create the gridded plot for theta_ref with 3x3 subplots
    # FIGURE 1: Policy's learned theta_ref parameter
    fig3, axes3 = plt.subplots(3, 3, figsize=(10, 10))
    # fig3.suptitle('Figure 1: Policy Learned $\\theta_{\\mathrm{ref}}$', fontsize=22, y=0.995)
    
    # Determine global color limits for consistent coloring
    vmin_theta_global = theta_ref_full_grid.min()
    vmax_theta_global = theta_ref_full_grid.max()
    
    if vmin_theta_global < 0.0 < vmax_theta_global:
        norm_theta_global = TwoSlopeNorm(vmin=vmin_theta_global, vcenter=0.0, vmax=vmax_theta_global)
        cmap_theta_global = 'RdYlBu_r'
    else:
        norm_theta_global = None
        cmap_theta_global = 'viridis'
    
    # Plot each (v, thetadot) combination in its own subplot
    for v_idx in range(grid_size_v):
        for td_idx in range(grid_size_thetadot):
            ax = axes3[td_idx, v_idx]
            
            # Get the data for this (v, thetadot) combination
            data = theta_ref_full_grid[v_idx, td_idx, :, :]
            
            # Plot using imshow with correct extent
            im = ax.imshow(data, cmap=cmap_theta_global, norm=norm_theta_global,
                          origin='lower', aspect='auto', interpolation='bilinear',
                          extent=[x_range[0], x_range[1], theta_range[0], theta_range[1]])
            
            # Add title with v and thetadot values
            ax.set_title(f'v={v_values[v_idx]:.1f} m/s, $\\dot{{\\theta}}$={thetadot_values[td_idx]:.1f} rad/s', fontsize=12)
            
            # Set custom x-axis ticks
            ax.set_xticks([-0.3, 0, 0.3])
            # Set custom y-axis ticks (5 values from -π to π)
            ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Set labels only on outer edges
            if td_idx == 2:  # Bottom row
                ax.set_xlabel('x (m)', fontsize=13)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', length=0)
            
            if v_idx == 0:  # Left column
                ax.set_ylabel('θ (rad)', fontsize=13)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', which='both', length=0)
    
    # Add a single colorbar for all subplots
    fig3.subplots_adjust(right=0.92)
    cbar_ax3 = fig3.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar3 = fig3.colorbar(im, cax=cbar_ax3)
    cbar3.set_label(r'$\theta_{\mathrm{ref}}$ (rad)', rotation=270, labelpad=20, fontsize=13)
    cbar3.ax.tick_params(labelsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    
    # Now create the gridded plot for MPC force with theta_ref=0 (naked MPC)
    # FIGURE 2: Baseline MPC force output (no learning)
    fig4, axes4 = plt.subplots(3, 3, figsize=(10, 10))
    # fig4.suptitle('Figure 2: MPC Force Output ($\\theta_{\\mathrm{ref}}=0$)', fontsize=22, y=0.995)
    
    # Determine global color limits for MPC force
    vmin_action_global = mpc_force_full_grid.min()
    vmax_action_global = mpc_force_full_grid.max()
    
    if vmin_action_global < 0.0 < vmax_action_global:
        norm_action_global = TwoSlopeNorm(vmin=vmin_action_global, vcenter=0.0, vmax=vmax_action_global)
        cmap_action_global = 'RdYlBu_r'
    else:
        norm_action_global = None
        cmap_action_global = 'viridis'
    
    # Plot each (v, thetadot) combination in its own subplot
    for v_idx in range(grid_size_v):
        for td_idx in range(grid_size_thetadot):
            ax = axes4[td_idx, v_idx]
            
            # Get the data for this (v, thetadot) combination
            data = mpc_force_full_grid[v_idx, td_idx, :, :]
            
            # Plot using imshow with correct extent
            im = ax.imshow(data, cmap=cmap_action_global, norm=norm_action_global,
                          origin='lower', aspect='auto', interpolation='bilinear',
                          extent=[x_range[0], x_range[1], theta_range[0], theta_range[1]])
            
            # Add title with v and thetadot values
            ax.set_title(f'v={v_values[v_idx]:.1f} m/s, $\\dot{{\\theta}}$={thetadot_values[td_idx]:.1f} rad/s', fontsize=12)
            
            # Set custom x-axis ticks
            ax.set_xticks([-0.3, 0, 0.3])
            # Set custom y-axis ticks (5 values from -π to π)
            ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Set labels only on outer edges
            if td_idx == 2:  # Bottom row
                ax.set_xlabel('x (m)', fontsize=13)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', length=0)
            
            if v_idx == 0:  # Left column
                ax.set_ylabel('θ (rad)', fontsize=13)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', which='both', length=0)
    
    # Add a single colorbar for all subplots
    fig4.subplots_adjust(right=0.92)
    cbar_ax4 = fig4.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar4 = fig4.colorbar(im, cax=cbar_ax4)
    cbar4.set_label(r'$F_{\mathrm{MPC}}$ (N, $\theta_{\mathrm{ref}}=0$)', rotation=270, labelpad=20, fontsize=13)
    cbar4.ax.tick_params(labelsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    
    # ========== Figure 5: Critic Q-value grid ==========
    # FIGURE 3: Critic Q-values
    if critic is not None:
        fig5, axes5 = plt.subplots(3, 3, figsize=(10, 10))
        # fig5.suptitle('Figure 3: Critic Q-values', fontsize=22, y=0.995)
        
        # Determine global color limits for critic Q-values
        vmin_critic_global = critic_full_grid.min()
        vmax_critic_global = critic_full_grid.max()
        
        if vmin_critic_global < 0.0 < vmax_critic_global:
            norm_critic_global = TwoSlopeNorm(vmin=vmin_critic_global, vcenter=0.0, vmax=vmax_critic_global)
            cmap_critic_global = 'RdYlBu_r'
        else:
            norm_critic_global = None
            cmap_critic_global = 'viridis'
        
        # Plot each (v, thetadot) combination in its own subplot
        for v_idx in range(grid_size_v):
            for td_idx in range(grid_size_thetadot):
                ax = axes5[td_idx, v_idx]
                
                # Get the data for this (v, thetadot) combination
                data = critic_full_grid[v_idx, td_idx, :, :]
                
                # Plot using imshow with correct extent
                im = ax.imshow(data, cmap=cmap_critic_global, norm=norm_critic_global,
                              origin='lower', aspect='auto', interpolation='bilinear',
                              extent=[x_range[0], x_range[1], theta_range[0], theta_range[1]])
                
                # Add title with v and thetadot values
                ax.set_title(f'v={v_values[v_idx]:.1f} m/s, $\\dot{{\\theta}}$={thetadot_values[td_idx]:.1f} rad/s', fontsize=12)
                
                # Set custom x-axis ticks
                ax.set_xticks([-0.3, 0, 0.3])
                # Set custom y-axis ticks (5 values from -π to π)
                ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
                ax.tick_params(axis='both', which='major', labelsize=12)
                
                # Set labels only on outer edges
                if td_idx == 2:  # Bottom row
                    ax.set_xlabel('x (m)', fontsize=13)
                else:
                    ax.set_xticklabels([])
                    ax.tick_params(axis='x', which='both', length=0)
                
                if v_idx == 0:  # Left column
                    ax.set_ylabel('θ (rad)', fontsize=13)
                else:
                    ax.set_yticklabels([])
                    ax.tick_params(axis='y', which='both', length=0)
        
        # Add a single colorbar for all subplots
        fig5.subplots_adjust(right=0.92)
        cbar_ax5 = fig5.add_axes([0.94, 0.15, 0.02, 0.7])
        cbar5 = fig5.colorbar(im, cax=cbar_ax5)
        cbar5.set_label(r'$Q$-value', rotation=270, labelpad=20, fontsize=13)
        cbar5.ax.tick_params(labelsize=12)
        
        plt.tight_layout(rect=[0, 0, 0.92, 1])
    
    # ========== Figure 6: Policy force grid ==========
    # FIGURE 4: Policy force output (with learned theta_ref)
    fig6, axes6 = plt.subplots(3, 3, figsize=(10, 10))
    # fig6.suptitle('Figure 4: Policy Force Output (with learned $\\theta_{\\mathrm{ref}}$)', fontsize=22, y=0.995)
    
    # Determine global color limits for policy force
    vmin_mpc_global = action_full_grid.min()
    vmax_mpc_global = action_full_grid.max()
    
    if vmin_mpc_global < 0.0 < vmax_mpc_global:
        norm_mpc_global = TwoSlopeNorm(vmin=vmin_mpc_global, vcenter=0.0, vmax=vmax_mpc_global)
        cmap_mpc_global = 'RdYlBu_r'
    else:
        norm_mpc_global = None
        cmap_mpc_global = 'viridis'
    
    # Plot each (v, thetadot) combination in its own subplot
    for v_idx in range(grid_size_v):
        for td_idx in range(grid_size_thetadot):
            ax = axes6[td_idx, v_idx]
            
            # Get the data for this (v, thetadot) combination
            data = action_full_grid[v_idx, td_idx, :, :]
            
            # Plot using imshow with correct extent
            im = ax.imshow(data, cmap=cmap_mpc_global, norm=norm_mpc_global,
                          origin='lower', aspect='auto', interpolation='bilinear',
                          extent=[x_range[0], x_range[1], theta_range[0], theta_range[1]])
            
            # Add title with v and thetadot values
            ax.set_title(f'v={v_values[v_idx]:.1f} m/s, $\\dot{{\\theta}}$={thetadot_values[td_idx]:.1f} rad/s', fontsize=12)
            
            # Set custom x-axis ticks
            ax.set_xticks([-0.3, 0, 0.3])
            # Set custom y-axis ticks (5 values from -π to π)
            ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Set labels only on outer edges
            if td_idx == 2:  # Bottom row
                ax.set_xlabel('x (m)', fontsize=13)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', length=0)
            
            if v_idx == 0:  # Left column
                ax.set_ylabel('θ (rad)', fontsize=13)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', which='both', length=0)
    
    # Add a single colorbar for all subplots
    fig6.subplots_adjust(right=0.92)
    cbar_ax6 = fig6.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar6 = fig6.colorbar(im, cax=cbar_ax6)
    cbar6.set_label(r'$F_{\mathrm{policy}}$ (N)', rotation=270, labelpad=20, fontsize=13)
    cbar6.ax.tick_params(labelsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    
    # Auto-save figures to PDF in the same directory as the actor checkpoint
    actor_dir = Path(actor_path).parent
    
    # Save theta_ref grid
    ep_suffix = f"_ep{episode_num}" if episode_num is not None else ""
    save_path_theta_grid = actor_dir / f"policy_heatmap_theta_ref_grid{ep_suffix}.pdf"
    try:
        fig3.savefig(save_path_theta_grid, bbox_inches='tight', dpi=150)
        print(f"Saved theta_ref grid plot to {save_path_theta_grid}")
    except Exception as e:
        print(f"Failed to save theta_ref grid plot: {e}")
    
    # Save MPC force grid (baseline, doesn't change with episodes)
    save_path_mpc_force_grid = actor_dir / f"policy_heatmap_mpc_force_grid.pdf"
    try:
        fig4.savefig(save_path_mpc_force_grid, bbox_inches='tight', dpi=150)
        print(f"Saved MPC force (theta_ref=0) grid plot to {save_path_mpc_force_grid}")
    except Exception as e:
        print(f"Failed to save MPC force grid plot: {e}")
    
    # Save critic grid
    if critic is not None:
        save_path_critic_grid = actor_dir / f"policy_heatmap_critic_grid{ep_suffix}.pdf"
        try:
            fig5.savefig(save_path_critic_grid, bbox_inches='tight', dpi=150)
            print(f"Saved critic grid plot to {save_path_critic_grid}")
        except Exception as e:
            print(f"Failed to save critic grid plot: {e}")
    
    # Save policy force grid (learned, changes with episodes)
    save_path_policy_force_grid = actor_dir / f"policy_heatmap_force_grid{ep_suffix}.pdf"
    try:
        fig6.savefig(save_path_policy_force_grid, bbox_inches='tight', dpi=150)
        print(f"Saved policy force grid plot to {save_path_policy_force_grid}")
    except Exception as e:
        print(f"Failed to save policy force grid plot: {e}")
    
    # Save figures if additional save_path requested
    if save_path is not None:
        base_path = Path(save_path)
        parent = base_path.parent
        stem = base_path.stem
        suffix = base_path.suffix
        
        # Save theta_ref grid
        save_path_theta_grid = parent / f"{stem}_theta_ref_grid{suffix}"
        try:
            fig3.savefig(save_path_theta_grid, bbox_inches='tight', dpi=150)
            print(f"Saved theta_ref grid plot to {save_path_theta_grid}")
        except Exception as e:
            print(f"Failed to save theta_ref grid plot: {e}")
        
        # Save force grid
        save_path_force_grid = parent / f"{stem}_force_grid{suffix}"
        try:
            fig4.savefig(save_path_force_grid, bbox_inches='tight', dpi=150)
            print(f"Saved force grid plot to {save_path_force_grid}")
        except Exception as e:
            print(f"Failed to save force grid plot: {e}")
    
    if plt_show:
        plt.show()


def plot_sac_policy_heatmap(
    actor_path: str,
    v_fixed: float = 0.0,
    thetadot_fixed: float = 0.0,
    x_range: tuple[float, float] = (-0.39, 0.39),
    theta_range: tuple[float, float] = (-np.pi, np.pi),
    resolution: int = 50,
    plt_show: bool = True,
    save_path: str | None = None,
    episode_num: int | None = None
):
    """Create heatmap grids showing force and critic outputs from vanilla SAC policy.
    
    Args:
        actor_path: Path to actor.pth checkpoint file
        v_fixed: Fixed cart velocity value (m/s)
        thetadot_fixed: Fixed pole angular velocity (rad/s)
        x_range: Range of cart positions to plot (min, max) in meters
        theta_range: Range of pole angles to plot (min, max) in radians
        resolution: Number of grid points along each axis
        plt_show: Whether to display the plot
        save_path: Optional path to save the figure
        episode_num: Episode number for filename
    """
    try:
        import torch
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
        from pathlib import Path
    except ImportError as e:
        print(f"Required libraries not available: {e}")
        return
    
    # Import required modules
    try:
        from leap_c.torch.nn.extractor import get_extractor_cls
        from leap_c.torch.rl.sac import SacActor, SacCritic
        import gymnasium as gym
    except ImportError as e:
        print(f"Failed to import required modules: {e}")
        return
    
    # Setup device and config
    device = "cpu"
    
    # Define observation space (same as in real_sac.py)
    _x_thr = x_range[1]
    obs_low = np.array([-_x_thr, -np.pi, -5, -21], dtype=np.float32)
    obs_high = np.array([_x_thr, np.pi, 5, 21], dtype=np.float32)
    obs_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
    action_space = gym.spaces.Box(low=np.array([-20.0]), high=np.array([20.0]), dtype=np.float32)
    
    # Initialize actor (no layer norm for actor in SAC)
    from leap_c.torch.nn.mlp import MlpConfig
    actor_mlp_cfg = MlpConfig(
        hidden_dims=(256, 256, 256),
        activation="relu",
        norm_layer=None,  # No layer norm for actor
    )
    
    extractor_cls = get_extractor_cls("identity")
    actor = SacActor(
        extractor_cls=extractor_cls,
        observation_space=obs_space,
        action_space=action_space,
        mlp_cfg=actor_mlp_cfg,
        distribution_name="squashed_gaussian",
    ).to(device)
    
    # Load actor weights
    try:
        actor.load_state_dict(torch.load(actor_path, map_location=device))
        actor.eval()
        print(f"Loaded actor from {actor_path}")
    except Exception as e:
        print(f"Failed to load actor: {e}")
        return
    
    # Load critic (with layer norm)
    critic_path = str(Path(actor_path).parent / "critic.pth")
    critic_mlp_cfg = MlpConfig(
        hidden_dims=(256, 256, 256),
        activation="relu",
        norm_layer="layer_norm",  # Layer norm for critic
    )
    
    critic = SacCritic(
        extractor_cls=extractor_cls,
        observation_space=obs_space,
        action_space=action_space,
        mlp_cfg=critic_mlp_cfg,
        num_critics=8,  # Match the training config
    ).to(device)
    
    try:
        critic.load_state_dict(torch.load(critic_path, map_location=device))
        critic.eval()
        print(f"Loaded critic from {critic_path}")
    except Exception as e:
        print(f"Warning: Failed to load critic: {e}")
        import traceback
        traceback.print_exc()
        critic = None
    
    # Create meshgrid for different (v, thetadot) combinations
    v_values = np.array([-1.0, 0.0, 1.0])
    thetadot_values = np.array([-5.0, 0.0, 5.0])
    
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    theta_vals = np.linspace(theta_range[0], theta_range[1], resolution)
    X, Theta = np.meshgrid(x_vals, theta_vals)
    
    grid_size_v = len(v_values)
    grid_size_thetadot = len(thetadot_values)
    
    force_full_grid = np.zeros((grid_size_v, grid_size_thetadot, resolution, resolution))
    critic_full_grid = np.zeros((grid_size_v, grid_size_thetadot, resolution, resolution))
    
    print("Evaluating SAC policy on grid with varying v and thetadot...")
    
    with torch.no_grad():
        for v_idx, v_val in enumerate(v_values):
            for td_idx, thetadot_val in enumerate(thetadot_values):
                for i in range(resolution):
                    for j in range(resolution):
                        state = np.array([X[i, j], Theta[i, j], v_val, thetadot_val], dtype=np.float32)
                        obs = torch.from_numpy(state).float().unsqueeze(0).to(device)
                        
                        # Get deterministic action from actor
                        action, _, _ = actor(obs, deterministic=True)
                        force_full_grid[v_idx, td_idx, i, j] = action[0].cpu().numpy()
                        
                        # Get critic Q-value if available
                        if critic is not None:
                            q_values = critic(obs, action)
                            q_value = torch.stack(q_values).min().cpu().numpy()
                            critic_full_grid[v_idx, td_idx, i, j] = q_value
    
    # Filter out inf and nan values
    force_full_grid = np.nan_to_num(force_full_grid, nan=0.0, posinf=0.0, neginf=0.0)
    critic_full_grid = np.nan_to_num(critic_full_grid, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\n=== SAC Force Analysis ===")
    print(f"Force range: [{force_full_grid.min():.4f}, {force_full_grid.max():.4f}]")
    print(f"Force mean: {force_full_grid.mean():.4f}, std: {force_full_grid.std():.4f}")
    
    # Clip to reasonable ranges
    force_full_grid = np.clip(force_full_grid, -100, 100)
    critic_full_grid = np.clip(critic_full_grid, -1000, 1000)
    
    # FIGURE 1: Force output grid (3x3 subplots)
    fig1, axes1 = plt.subplots(3, 3, figsize=(10, 10))
    
    vmin_force = force_full_grid.min()
    vmax_force = force_full_grid.max()
    
    if vmin_force < 0.0 < vmax_force:
        norm_force = TwoSlopeNorm(vmin=vmin_force, vcenter=0.0, vmax=vmax_force)
        cmap_force = 'RdYlBu_r'
    else:
        norm_force = None
        cmap_force = 'viridis'
    
    for v_idx in range(grid_size_v):
        for td_idx in range(grid_size_thetadot):
            ax = axes1[td_idx, v_idx]
            data = force_full_grid[v_idx, td_idx, :, :]
            
            im = ax.imshow(data, cmap=cmap_force, norm=norm_force,
                          origin='lower', aspect='auto', interpolation='bilinear',
                          extent=[x_range[0], x_range[1], theta_range[0], theta_range[1]])
            
            ax.set_title(f'v={v_values[v_idx]:.1f} m/s, $\\dot{{\\theta}}$={thetadot_values[td_idx]:.1f} rad/s', fontsize=12)
            ax.set_xticks([-0.3, 0, 0.3])
            ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            if td_idx == 2:
                ax.set_xlabel('x (m)', fontsize=13)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', length=0)
            
            if v_idx == 0:
                ax.set_ylabel('θ (rad)', fontsize=13)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', which='both', length=0)
    
    fig1.subplots_adjust(right=0.92)
    cbar_ax1 = fig1.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar1 = fig1.colorbar(im, cax=cbar_ax1)
    cbar1.set_label('Force (N)', rotation=270, labelpad=20, fontsize=13)
    cbar1.ax.tick_params(labelsize=12)
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    
    # FIGURE 2: Critic Q-value grid (3x3 subplots)
    if critic is not None:
        fig2, axes2 = plt.subplots(3, 3, figsize=(10, 10))
        
        vmin_critic = critic_full_grid.min()
        vmax_critic = critic_full_grid.max()
        
        if vmin_critic < 0.0 < vmax_critic:
            norm_critic = TwoSlopeNorm(vmin=vmin_critic, vcenter=0.0, vmax=vmax_critic)
            cmap_critic = 'RdYlBu_r'
        else:
            norm_critic = None
            cmap_critic = 'viridis'
        
        for v_idx in range(grid_size_v):
            for td_idx in range(grid_size_thetadot):
                ax = axes2[td_idx, v_idx]
                data = critic_full_grid[v_idx, td_idx, :, :]
                
                im = ax.imshow(data, cmap=cmap_critic, norm=norm_critic,
                              origin='lower', aspect='auto', interpolation='bilinear',
                              extent=[x_range[0], x_range[1], theta_range[0], theta_range[1]])
                
                ax.set_title(f'v={v_values[v_idx]:.1f} m/s, $\\dot{{\\theta}}$={thetadot_values[td_idx]:.1f} rad/s', fontsize=12)
                ax.set_xticks([-0.3, 0, 0.3])
                ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
                ax.tick_params(axis='both', which='major', labelsize=12)
                
                if td_idx == 2:
                    ax.set_xlabel('x (m)', fontsize=13)
                else:
                    ax.set_xticklabels([])
                    ax.tick_params(axis='x', which='both', length=0)
                
                if v_idx == 0:
                    ax.set_ylabel('θ (rad)', fontsize=13)
                else:
                    ax.set_yticklabels([])
                    ax.tick_params(axis='y', which='both', length=0)
        
        fig2.subplots_adjust(right=0.92)
        cbar_ax2 = fig2.add_axes([0.94, 0.15, 0.02, 0.7])
        cbar2 = fig2.colorbar(im, cax=cbar_ax2)
        cbar2.set_label('Q-value', rotation=270, labelpad=20, fontsize=13)
        cbar2.ax.tick_params(labelsize=12)
        plt.tight_layout(rect=[0, 0, 0.92, 1])
    
    # Save figures
    if save_path or episode_num is not None:
        parent = Path(actor_path).parent
        stem = "policy_heatmap"
        suffix = f"_ep{episode_num}.pdf" if episode_num is not None else ".pdf"
        
        save_path_force_grid = parent / f"{stem}_force_grid{suffix}"
        try:
            fig1.savefig(save_path_force_grid, bbox_inches='tight', dpi=150)
            print(f"Saved force grid plot to {save_path_force_grid}")
        except Exception as e:
            print(f"Failed to save force grid plot: {e}")
        
        if critic is not None:
            save_path_critic_grid = parent / f"{stem}_critic_grid{suffix}"
            try:
                fig2.savefig(save_path_critic_grid, bbox_inches='tight', dpi=150)
                print(f"Saved critic grid plot to {save_path_critic_grid}")
            except Exception as e:
                print(f"Failed to save critic grid plot: {e}")
        else:
            print(f"Skipping critic plot - critic not loaded successfully")
    
    if plt_show:
        plt.show()


def plot_critic_heatmap(
    critic_path: str,
    x_range: tuple = (-0.35, 0.35),
    theta_range: tuple = (-np.pi, np.pi),
    resolution: int = 50,
    plt_show: bool = False,
    save_path: str = None,
    episode_num: int = None
):
    """
    Generate Q-value heatmap grid (3x3) from trained critic network.
    Shows Q-values for different (v, thetadot) combinations.
    
    Args:
        critic_path: Path to critic.pth checkpoint file
        x_range: Range of cart positions to visualize (min, max) in meters
        theta_range: Range of pole angles to visualize (min, max) in radians
        resolution: Grid resolution (number of points per dimension)
        plt_show: Whether to display plots interactively
        save_path: Base path for saving PDFs (if None, uses checkpoint directory)
        episode_num: Episode number for filename (optional)
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import os
    from pathlib import Path
    import gymnasium as gym
    
    # Import necessary modules
    from bachelor.acados_cartpole.real_sac_zop.my_sac import SacCritic
    from bachelor.acados_cartpole.real_sac_zop.my_sac_zop import SacZopTrainerConfig
    from bachelor.acados_cartpole.my_planner import (
        CartPolePlannerConfig,
        CartPolePlanner,
        create_custom_cartpole_params
    )
    from leap_c.planner import ControllerFromPlanner
    from leap_c.torch.nn.extractor import get_extractor_cls
    
    device = "cpu"
    
    # Load config (same as training)
    cfg_saczop = SacZopTrainerConfig()
    cfg_saczop.critic_mlp.norm_layer = "layer_norm"  # match training config
    
    # Setup planner and action space (same as training)
    cfg_planner = CartPolePlannerConfig()
    params = create_custom_cartpole_params("global", cfg_planner.N_horizon)
    planner = CartPolePlanner(cfg_planner, params)
    controller_wrapped = ControllerFromPlanner(planner)
    
    # Define observation and action spaces (same as training)
    _x_thr = getattr(cfg_planner, "x_threshold", 0.4)
    _x_low = -float(_x_thr)
    _x_high = float(_x_thr)
    
    obs_low = np.array([_x_low, -np.pi, -5, -21], dtype=np.float32)
    obs_high = np.array([_x_high, np.pi, 5, 21], dtype=np.float32)
    obs_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
    action_space = controller_wrapped.param_space
    
    # Load critic network with proper config
    extractor_cls = get_extractor_cls("identity")
    critic = SacCritic(
        extractor_cls=extractor_cls,
        observation_space=obs_space,
        action_space=action_space,
        mlp_cfg=cfg_saczop.critic_mlp,
        num_critics=cfg_saczop.num_critics,
    ).to(device)
    
    # Load checkpoint
    critic.load_state_dict(torch.load(critic_path, map_location=device))
    critic.eval()
    print(f"Loaded critic from {critic_path}")
    
    # Create meshgrid for different (v, thetadot) combinations
    v_values = np.array([-1.0, 0.0, 1.0])
    thetadot_values = np.array([-5.0, 0.0, 5.0])
    
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    theta_vals = np.linspace(theta_range[0], theta_range[1], resolution)
    X, Theta = np.meshgrid(x_vals, theta_vals)
    
    grid_size_v = len(v_values)
    grid_size_thetadot = len(thetadot_values)
    
    critic_full_grid = np.zeros((grid_size_v, grid_size_thetadot, resolution, resolution))
    
    print(f"Computing Q-values over {resolution}x{resolution} grid for 9 (v, thetadot) combinations...")
    
    # For each grid point, we need a parameter to evaluate Q(s, param)
    # We'll use a zero parameter as a reference point
    param_dim = int(np.prod(action_space.shape))
    param_ref = torch.zeros(1, param_dim, device=device)
    
    # Compute Q-values for all (v, thetadot) combinations
    with torch.no_grad():
        for v_idx, v_val in enumerate(v_values):
            for td_idx, thetadot_val in enumerate(thetadot_values):
                for i in range(resolution):
                    for j in range(resolution):
                        x = X[i, j]
                        theta = Theta[i, j]
                        
                        # Create state tensor: [x, theta, v, thetadot]
                        state = torch.tensor([[x, theta, v_val, thetadot_val]], 
                                            dtype=torch.float32, device=device)
                        
                        # Get Q-values from both critics
                        q_values = critic(state, param_ref)
                        q_values_cat = torch.cat(q_values, dim=1)
                        
                        # Take minimum Q-value (standard SAC practice)
                        q_min = torch.min(q_values_cat, dim=1).values
                        critic_full_grid[v_idx, td_idx, i, j] = q_min.item()
    
    # Filter out inf and nan values
    critic_full_grid = np.nan_to_num(critic_full_grid, nan=0.0, posinf=0.0, neginf=0.0)
    critic_full_grid = np.clip(critic_full_grid, -1000, 1000)
    
    print(f"Q-value range: [{critic_full_grid.min():.2f}, {critic_full_grid.max():.2f}]")
    
    # ========== Create 3x3 grid plot ==========
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    
    # Determine global color limits for critic Q-values
    vmin_critic_global = critic_full_grid.min()
    vmax_critic_global = critic_full_grid.max()
    
    if vmin_critic_global < 0.0 < vmax_critic_global:
        norm_critic_global = TwoSlopeNorm(vmin=vmin_critic_global, vcenter=0.0, vmax=vmax_critic_global)
        cmap_critic_global = 'RdYlBu_r'
    else:
        norm_critic_global = None
        cmap_critic_global = 'viridis'
    
    # Plot each (v, thetadot) combination in its own subplot
    for v_idx in range(grid_size_v):
        for td_idx in range(grid_size_thetadot):
            ax = axes[td_idx, v_idx]
            
            # Get the data for this (v, thetadot) combination
            data = critic_full_grid[v_idx, td_idx, :, :]
            
            # Plot using imshow with correct extent
            im = ax.imshow(data, cmap=cmap_critic_global, norm=norm_critic_global,
                          origin='lower', aspect='auto', interpolation='bilinear',
                          extent=[x_range[0], x_range[1], theta_range[0], theta_range[1]])
            
            # Add title with v and thetadot values
            ax.set_title(f'v={v_values[v_idx]:.1f} m/s, $\\dot{{\\theta}}$={thetadot_values[td_idx]:.1f} rad/s', fontsize=12)
            
            # Set custom x-axis ticks
            ax.set_xticks([-0.3, 0, 0.3])
            # Set custom y-axis ticks (7 values from -π to π)
            ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Set labels only on outer edges
            if td_idx == 2:  # Bottom row
                ax.set_xlabel('x (m)', fontsize=13)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', length=0)
            
            if v_idx == 0:  # Left column
                ax.set_ylabel('θ (rad)', fontsize=13)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', which='both', length=0)
    
    # Add a single colorbar for all subplots
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'$Q$-value', rotation=270, labelpad=20, fontsize=13)
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    
    # Determine save directory
    if save_path is None:
        checkpoint_dir = os.path.dirname(critic_path)
        save_dir = checkpoint_dir
    else:
        save_dir = os.path.dirname(save_path) if save_path else '.'
    
    # Create filename suffix
    ep_suffix = f"_ep{episode_num}" if episode_num is not None else ""
    
    # Save figure
    output_path = os.path.join(save_dir, f'critic_heatmap_grid{ep_suffix}.pdf')
    try:
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Saved critic grid plot to {output_path}")
    except Exception as e:
        print(f"Failed to save critic grid plot: {e}")
    
    if plt_show:
        plt.show()
    else:
        plt.close()
    
    print("\n✓ Critic heatmap generation complete!")
    return critic_full_grid




