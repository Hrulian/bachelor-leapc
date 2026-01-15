import os
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# Directory containing the trajectory CSV files
TRAJECTORIES_DIR = os.path.join(os.path.dirname(__file__), "Trajectories")

def read_trajectory_csv(filepath):
    """Read a trajectory CSV file and return time, x position, and reward data.
    
    Returns:
        tuple: (times, x_positions, rewards) as lists of floats
    """
    times = []
    x_positions = []
    rewards = []
    
    try:
        with open(filepath, 'r', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    t = float(row.get('t', ''))
                    x = float(row.get('x_m', ''))
                    r = float(row.get('accumulated_reward', ''))
                    times.append(t)
                    x_positions.append(x)
                    rewards.append(r)
                except (ValueError, TypeError):
                    continue
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None, None
    
    return times, x_positions, rewards


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
    
    import numpy as np
    
    # Collect data
    mpc_data = []
    sac_data = []
    mpc_reward_data = []
    sac_reward_data = []
    
    for filepath in mpc_files:
        times, x_positions, rewards = read_trajectory_csv(filepath)
        if times and x_positions:
            mpc_data.append((times, x_positions))
            mpc_reward_data.append((times, rewards))
    
    for filepath in sac_files:
        times, x_positions, rewards = read_trajectory_csv(filepath)
        if times and x_positions:
            sac_data.append((times, x_positions))
            sac_reward_data.append((times, rewards))
    
    # Plot 1: Stacked position trajectories (MPC and SAC-ZOP)
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    
    # Plot MPC trajectories
    for i, (times, x_positions) in enumerate(mpc_data):
        # Filter data to first 20 seconds
        mask = np.array(times) <= 20.0
        times_filtered = np.array(times)[mask]
        x_filtered = np.array(x_positions)[mask]
        ax1.plot(times_filtered, x_filtered, color='red', alpha=0.6, linewidth=1.8)
    
    # Configure MPC position plot
    ax1.set_ylabel('Cart Position x (m)', fontsize=12)
    ax1.grid(True, alpha=0.3, which='major')
    ax1.grid(True, alpha=0.15, which='minor', linestyle=':')
    ax1.minorticks_on()
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1.2)
    
    # Plot SAC trajectories
    for i, (times, x_positions) in enumerate(sac_data):
        # Filter data to first 20 seconds
        mask = np.array(times) <= 20.0
        times_filtered = np.array(times)[mask]
        x_filtered = np.array(x_positions)[mask]
        ax2.plot(times_filtered, x_filtered, color='blue', alpha=0.6, linewidth=1.8)
    
    # Configure SAC position plot
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Cart Position x (m)', fontsize=12)
    ax2.grid(True, alpha=0.3, which='major')
    ax2.grid(True, alpha=0.15, which='minor', linestyle=':')
    ax2.minorticks_on()
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1.2)
    
    # Set x-axis limit to 20 seconds
    ax2.set_xlim(0, 20)
    
    plt.tight_layout()
    
    # Save stacked position plot as PDF
    save_path1 = os.path.join(TRAJECTORIES_DIR, 'position_comparison.pdf')
    try:
        fig1.savefig(save_path1, dpi=150, bbox_inches='tight')
        print(f"Saved position comparison plot to: {save_path1}")
    except Exception as e:
        print(f"Failed to save position comparison plot: {e}")
    
    # Plot 2: Accumulated Reward (with mean lines)
    fig2, ax3 = plt.subplots(figsize=(12, 6))
    
    # Plot MPC rewards
    for i, (times, rewards) in enumerate(mpc_reward_data):
        # Filter data to first 20 seconds
        mask = np.array(times) <= 20.0
        times_filtered = np.array(times)[mask]
        rewards_filtered = np.array(rewards)[mask]
        ax3.plot(times_filtered, rewards_filtered, color='red', alpha=0.6, linewidth=1.8,
                label='MPC' if i == 0 else '')
    
    # Plot SAC rewards
    for i, (times, rewards) in enumerate(sac_reward_data):
        # Filter data to first 20 seconds
        mask = np.array(times) <= 20.0
        times_filtered = np.array(times)[mask]
        rewards_filtered = np.array(rewards)[mask]
        ax3.plot(times_filtered, rewards_filtered, color='blue', alpha=0.6, linewidth=1.8,
                label='SAC-ZOP' if i == 0 else '')
    
    # Compute and plot mean rewards
    if mpc_reward_data:
        # Get data up to 20 seconds
        mpc_20s = []
        for times, rewards in mpc_reward_data:
            mask = np.array(times) <= 20.0
            mpc_20s.append((np.array(times)[mask], np.array(rewards)[mask]))
        
        min_len = min(len(r) for _, r in mpc_20s)
        mpc_reward_mean = np.mean([r[:min_len] for _, r in mpc_20s], axis=0)
        ref_times = mpc_20s[0][0][:min_len]
        ax3.plot(ref_times, mpc_reward_mean, color='darkred', linewidth=4.5,
                label='MPC (mean)', linestyle='--')
    
    if sac_reward_data:
        # Get data up to 20 seconds
        sac_20s = []
        for times, rewards in sac_reward_data:
            mask = np.array(times) <= 20.0
            sac_20s.append((np.array(times)[mask], np.array(rewards)[mask]))
        
        min_len = min(len(r) for _, r in sac_20s)
        sac_reward_mean = np.mean([r[:min_len] for _, r in sac_20s], axis=0)
        ref_times = sac_20s[0][0][:min_len]
        ax3.plot(ref_times, sac_reward_mean, color='darkblue', linewidth=4.5,
                label='SAC-ZOP (mean)', linestyle='--')
    
    # Configure reward plot
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Accumulated Reward', fontsize=12)
    ax3.grid(True, alpha=0.3, which='major')
    ax3.grid(True, alpha=0.15, which='minor', linestyle=':')
    ax3.minorticks_on()
    ax3.legend(fontsize=11, loc='best')
    ax3.set_xlim(0, 20)
    
    plt.tight_layout()
    
    # Save reward plot as PDF
    save_path2 = os.path.join(TRAJECTORIES_DIR, 'reward_comparison.pdf')
    try:
        fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
        print(f"Saved reward plot to: {save_path2}")
    except Exception as e:
        print(f"Failed to save reward plot: {e}")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    plot_all_trajectories()
