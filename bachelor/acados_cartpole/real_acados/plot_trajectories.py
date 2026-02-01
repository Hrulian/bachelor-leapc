import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from bachelor.acados_cartpole.my_helpers import reward_eval

# Directory containing the trajectory CSV files
TRAJECTORIES_DIR = os.path.join(os.path.dirname(__file__), "Trajectories")

def read_trajectory_csv(filepath):
    """Read a trajectory CSV file and return time, x position, theta angle, and computed reward data.
    
    Returns:
        tuple: (times, x_positions, thetas, rewards) as lists of floats
    """
    times = []
    x_positions = []
    thetas = []
    rewards = []
    
    try:
        with open(filepath, 'r', newline='') as fh:
            reader = csv.DictReader(fh)
            accumulated_reward = 0.0
            for row in reader:
                try:
                    t = float(row.get('t', ''))
                    x_m = float(row.get('x_m', ''))
                    theta = float(row.get('theta_unwrapped', ''))
                    
                    # Calculate reward using reward_eval function
                    # Need to convert x_m back to counts for reward_eval
                    x_counts = x_m / 0.001  # inverse of counts_to_meters
                    step_reward = reward_eval(x_counts)
                    accumulated_reward += step_reward
                    
                    times.append(t)
                    x_positions.append(x_m)
                    thetas.append(theta)
                    rewards.append(accumulated_reward)
                except (ValueError, TypeError, KeyError):
                    continue
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None, None, None
    
    return times, x_positions, thetas, rewards


def plot_all_trajectories():
    """Plot all MPC and SAC trajectories from the Trajectories folder."""
    
    if not os.path.exists(TRAJECTORIES_DIR):
        print(f"ERROR: Trajectories directory not found at {TRAJECTORIES_DIR}")
        return
    
    # Find all matching CSV files
    mpc_files = []
    sac_files = []
    
    for filename in os.listdir(TRAJECTORIES_DIR):
        if filename.startswith("eval_mpc_50ms_") and filename.endswith(".csv"):
            mpc_files.append(os.path.join(TRAJECTORIES_DIR, filename))
        elif filename.startswith("eval_sac_50ms_") and filename.endswith(".csv"):
            sac_files.append(os.path.join(TRAJECTORIES_DIR, filename))
    
    if not mpc_files and not sac_files:
        print("No trajectory files found matching pattern 'eval_mpc_50ms_*' or 'eval_sac_50ms_*'")
        return
    
    print(f"Found {len(mpc_files)} MPC trajectories and {len(sac_files)} SAC trajectories")
    
    # Collect data
    mpc_data = []
    sac_data = []
    mpc_reward_data = []
    sac_reward_data = []
    
    for filepath in mpc_files:
        times, x_positions, thetas, rewards = read_trajectory_csv(filepath)
        if times and x_positions:
            mpc_data.append((times, x_positions, thetas))
            mpc_reward_data.append((times, rewards))
            print(f"  Loaded MPC: {os.path.basename(filepath)} ({len(times)} points)")
    
    for filepath in sac_files:
        times, x_positions, thetas, rewards = read_trajectory_csv(filepath)
        if times and x_positions:
            sac_data.append((times, x_positions, thetas))
            sac_reward_data.append((times, rewards))
            print(f"  Loaded SAC: {os.path.basename(filepath)} ({len(times)} points)")
    
    # Compute global axis limits for position and angle plots
    all_x = []
    all_theta = []
    for times, x_positions, thetas in mpc_data + sac_data:
        all_x.extend(x_positions)
        all_theta.extend(thetas)
    
    x_min, x_max = min(all_x), max(all_x)
    theta_min, theta_max = min(all_theta), max(all_theta)
    
    # Add 10% padding to limits
    x_padding = (x_max - x_min) * 0.1
    theta_padding = (theta_max - theta_min) * 0.1
    x_lim = (x_min - x_padding, x_max + x_padding)
    theta_lim = (theta_min - theta_padding, theta_max + theta_padding)
    
    print(f"\nShared axis limits:")
    print(f"  Position: [{x_lim[0]:.3f}, {x_lim[1]:.3f}] m")
    print(f"  Angle: [{theta_lim[0]:.3f}, {theta_lim[1]:.3f}] rad")
    
    # Define colors (less intense)
    mpc_color = '#FF8C42'  # Soft orange
    sac_color = '#5EBA7D'  # Soft green
    
    # ========== Figure 1: Accumulated Reward Comparison ==========
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot MPC rewards
    for i, (times, rewards) in enumerate(mpc_reward_data):
        times_arr = np.array(times)
        rewards_arr = np.array(rewards)
        mask = times_arr <= 12.0
        times_filtered = times_arr[mask]
        rewards_filtered = rewards_arr[mask]
        ax1.plot(times_filtered, rewards_filtered, color=mpc_color, alpha=1.0, linewidth=1.5,
                label='MPC' if i == 0 else '')
    
    # Plot SAC rewards
    for i, (times, rewards) in enumerate(sac_reward_data):
        times_arr = np.array(times)
        rewards_arr = np.array(rewards)
        mask = times_arr <= 12.0
        times_filtered = times_arr[mask]
        rewards_filtered = rewards_arr[mask]
        ax1.plot(times_filtered, rewards_filtered, color=sac_color, alpha=1.0, linewidth=1.5,
                label='SAC-ZOP' if i == 0 else '')
    
    # Configure reward plot
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Accumulated Position Error (m)', fontsize=12)
    ax1.grid(True, alpha=0.3, which='major')
    ax1.grid(True, alpha=0.15, which='minor', linestyle=':')
    ax1.minorticks_on()
    ax1.legend(fontsize=11, loc='best')
    ax1.set_xlim(0, 12)
    
    plt.tight_layout()
    
    # Save reward plot
    save_path1 = os.path.join(TRAJECTORIES_DIR, 'reward_comparison.pdf')
    try:
        fig1.savefig(save_path1, format='pdf', dpi=150, bbox_inches='tight')
        print(f"\nSaved reward comparison plot to: {save_path1}")
    except Exception as e:
        print(f"Failed to save reward comparison plot: {e}")
    
    # ========== Figure 2: Combined Trajectories (MPC and SAC-ZOP) ==========
    fig2, ((ax2a, ax2b), (ax2c, ax2d)) = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
    
    # Plot MPC positions (top left)
    for i, (times, x_positions, thetas) in enumerate(mpc_data):
        times_arr = np.array(times)
        x_arr = np.array(x_positions)
        mask = times_arr <= 12.0
        times_filtered = times_arr[mask]
        x_filtered = x_arr[mask]
        ax2a.plot(times_filtered, x_filtered, color=mpc_color, alpha=1.0, linewidth=1.5,
                 label='MPC' if i == 0 else '')
    
    # Plot MPC angles (bottom left)
    for i, (times, x_positions, thetas) in enumerate(mpc_data):
        times_arr = np.array(times)
        theta_arr = np.array(thetas)
        mask = times_arr <= 12.0
        times_filtered = times_arr[mask]
        theta_filtered = theta_arr[mask]
        ax2c.plot(times_filtered, theta_filtered, color=mpc_color, alpha=1.0, linewidth=1.5,
                 label='MPC' if i == 0 else '')
    
    # Plot SAC positions (top right)
    for i, (times, x_positions, thetas) in enumerate(sac_data):
        times_arr = np.array(times)
        x_arr = np.array(x_positions)
        mask = times_arr <= 12.0
        times_filtered = times_arr[mask]
        x_filtered = x_arr[mask]
        ax2b.plot(times_filtered, x_filtered, color=sac_color, alpha=1.0, linewidth=1.5,
                 label='SAC-ZOP' if i == 0 else '')
    
    # Plot SAC angles (bottom right)
    for i, (times, x_positions, thetas) in enumerate(sac_data):
        times_arr = np.array(times)
        theta_arr = np.array(thetas)
        mask = times_arr <= 12.0
        times_filtered = times_arr[mask]
        theta_filtered = theta_arr[mask]
        ax2d.plot(times_filtered, theta_filtered, color=sac_color, alpha=1.0, linewidth=1.5,
                 label='SAC-ZOP' if i == 0 else '')
    
    # Configure MPC position subplot (top left)
    ax2a.set_ylabel('x (m)', fontsize=12)
    ax2a.grid(True, alpha=0.3, which='major')
    ax2a.grid(True, alpha=0.15, which='minor', linestyle=':')
    ax2a.minorticks_on()
    ax2a.legend(fontsize=11, loc='best')
    ax2a.set_title('MPC Position', fontsize=13)
    ax2a.set_ylim(x_lim)
    
    # Configure SAC position subplot (top right)
    ax2b.set_ylabel('x (m)', fontsize=12)
    ax2b.grid(True, alpha=0.3, which='major')
    ax2b.grid(True, alpha=0.15, which='minor', linestyle=':')
    ax2b.minorticks_on()
    ax2b.legend(fontsize=11, loc='best')
    ax2b.set_title('SAC-ZOP Position', fontsize=13)
    ax2b.set_ylim(x_lim)
    
    # Configure MPC angle subplot (bottom left)
    ax2c.set_xlabel('Time (s)', fontsize=12)
    ax2c.set_ylabel('θ (rad, unwrapped)', fontsize=12)
    ax2c.grid(True, alpha=0.3, which='major')
    ax2c.grid(True, alpha=0.15, which='minor', linestyle=':')
    ax2c.minorticks_on()
    ax2c.set_xlim(0, 12)
    ax2c.set_ylim(theta_lim)
    ax2c.set_title('MPC Angle', fontsize=13)
    
    # Configure SAC angle subplot (bottom right)
    ax2d.set_xlabel('Time (s)', fontsize=12)
    ax2d.set_ylabel('θ (rad, unwrapped)', fontsize=12)
    ax2d.grid(True, alpha=0.3, which='major')
    ax2d.grid(True, alpha=0.15, which='minor', linestyle=':')
    ax2d.minorticks_on()
    ax2d.set_xlim(0, 12)
    ax2d.set_ylim(theta_lim)
    ax2d.set_title('SAC-ZOP Angle', fontsize=13)
    
    plt.tight_layout()
    
    # Save combined trajectories plot
    save_path2 = os.path.join(TRAJECTORIES_DIR, 'trajectories_comparison.pdf')
    try:
        fig2.savefig(save_path2, format='pdf', dpi=150, bbox_inches='tight')
        print(f"Saved combined trajectories plot to: {save_path2}")
    except Exception as e:
        print(f"Failed to save combined trajectories plot: {e}")
    
    # Show all plots
    plt.show()


if __name__ == "__main__":
    plot_all_trajectories()
