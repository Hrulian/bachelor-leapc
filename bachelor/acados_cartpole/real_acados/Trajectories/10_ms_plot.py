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


def plot_zoomed_comparison():
    """Plot zoomed comparison: left shows position trajectories, right shows reward curves."""
    
    if not os.path.exists(TRAJECTORIES_DIR):
        print(f"ERROR: Trajectories directory not found at {TRAJECTORIES_DIR}")
        return
    
    # Find all matching CSV files
    mpc_files = []
    sac_files = []
    
    for filename in os.listdir(TRAJECTORIES_DIR):
        if filename.startswith("eval_mpc_") and filename.endswith(".csv"):
            mpc_files.append(os.path.join(TRAJECTORIES_DIR, filename))
        elif filename.startswith("eval_sac_") and filename.endswith(".csv"):
            sac_files.append(os.path.join(TRAJECTORIES_DIR, filename))
    
    if not mpc_files or not sac_files:
        print("Need at least one MPC and one SAC file!")
        return
    
    print(f"Found {len(mpc_files)} MPC trajectories and {len(sac_files)} SAC trajectories")
    
    # Take first file from each
    mpc_file = mpc_files[0]
    sac_file = sac_files[0]
    
    print(f"\nUsing for trajectory comparison:")
    print(f"  MPC: {os.path.basename(mpc_file)}")
    print(f"  SAC: {os.path.basename(sac_file)}")
    
    # Load data
    mpc_times, mpc_x, mpc_theta, mpc_rewards = read_trajectory_csv(mpc_file)
    sac_times, sac_x, sac_theta, sac_rewards = read_trajectory_csv(sac_file)
    
    # Convert to numpy arrays
    mpc_times = np.array(mpc_times)
    mpc_x = np.array(mpc_x)
    mpc_rewards = np.array(mpc_rewards)
    
    sac_times = np.array(sac_times)
    sac_x = np.array(sac_x)
    sac_rewards = np.array(sac_rewards)
    
    # Filter to time window [4, 10] seconds
    mpc_mask = (mpc_times >= 4.0) & (mpc_times <= 10.0)
    sac_mask = (sac_times >= 4.0) & (sac_times <= 10.0)
    
    mpc_times_zoom = mpc_times[mpc_mask]
    mpc_x_zoom = mpc_x[mpc_mask]
    mpc_rewards_zoom = mpc_rewards[mpc_mask]
    
    sac_times_zoom = sac_times[sac_mask]
    sac_x_zoom = sac_x[sac_mask]
    sac_rewards_zoom = sac_rewards[sac_mask]
    
    # Normalize rewards to start at zero (subtract first value)
    if len(mpc_rewards_zoom) > 0:
        mpc_rewards_zoom_normalized = mpc_rewards_zoom - mpc_rewards_zoom[0]
        print(f"\nMPC reward offset: {mpc_rewards_zoom[0]:.4f}")
    else:
        mpc_rewards_zoom_normalized = mpc_rewards_zoom
    
    if len(sac_rewards_zoom) > 0:
        sac_rewards_zoom_normalized = sac_rewards_zoom - sac_rewards_zoom[0]
        print(f"SAC reward offset: {sac_rewards_zoom[0]:.4f}")
    else:
        sac_rewards_zoom_normalized = sac_rewards_zoom
    
    print(f"\nZoomed data (4-10 seconds):")
    print(f"  MPC: {len(mpc_times_zoom)} points")
    print(f"  SAC: {len(sac_times_zoom)} points")
    
    # Define colors
    mpc_color = '#FF8C42'  # Soft orange
    sac_color = '#5EBA7D'  # Soft green
    
    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ========== Left plot: Position trajectories ==========
    ax1.plot(mpc_times_zoom, mpc_x_zoom, color=mpc_color, linewidth=2.0, 
             label='MPC', alpha=0.9)
    ax1.plot(sac_times_zoom, sac_x_zoom, color=sac_color, linewidth=2.0, 
             label='SAC-ZOP', alpha=0.9)
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('x (m)', fontsize=12)
    ax1.set_title('Cart Position', fontsize=13)
    ax1.grid(True, alpha=0.3, which='major')
    ax1.grid(True, alpha=0.15, which='minor', linestyle=':')
    ax1.minorticks_on()
    ax1.legend(fontsize=11, loc='best')
    ax1.set_xlim(4, 9)
    
    # ========== Right plot: Reward curves (normalized) ==========
    ax2.plot(mpc_times_zoom, mpc_rewards_zoom_normalized, color=mpc_color, linewidth=2.0, 
             label='MPC', alpha=0.9)
    ax2.plot(sac_times_zoom, sac_rewards_zoom_normalized, color=sac_color, linewidth=2.0, 
             label='SAC-ZOP', alpha=0.9)
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Accumulated Position Error (m)', fontsize=12)
    ax2.set_title('Accumulated Error', fontsize=13)
    ax2.grid(True, alpha=0.3, which='major')
    ax2.grid(True, alpha=0.15, which='minor', linestyle=':')
    ax2.minorticks_on()
    ax2.legend(fontsize=11, loc='best')
    ax2.set_xlim(4, 9)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(TRAJECTORIES_DIR, 'zoomed_comparison_7to8s.pdf')
    try:
        fig.savefig(save_path, format='pdf', dpi=150, bbox_inches='tight')
        print(f"\nSaved zoomed comparison plot to: {save_path}")
    except Exception as e:
        print(f"Failed to save plot: {e}")
    
    plt.show()


if __name__ == "__main__":
    plot_zoomed_comparison()