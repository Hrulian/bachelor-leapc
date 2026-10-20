import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib as mpl



def advanced_smooth(x_values, y_values, smoothing_param=0.6, viewport_scale=1000):
    """
    Apply advanced exponential smoothing with debias weighting.
    This matches the TensorBoard smoothing algorithm.
    
    Args:
        x_values: Array of x-axis values (episodes)
        y_values: Array of y-axis values (rewards)
        smoothing_param: Smoothing parameter (0-1, higher = more smoothing)
        viewport_scale: Scale factor for x-axis range normalization
    
    Returns:
        Array of smoothed y-values
    """
    if len(y_values) == 0:
        return y_values
    
    smoothing_weight = min(np.sqrt(smoothing_param), 0.999)
    
    last_y = 0
    debias_weight = 0
    smoothed = np.zeros_like(y_values, dtype=float)
    
    range_of_x = x_values[-1] - x_values[0] if len(x_values) > 1 else 1
    
    for i in range(len(y_values)):
        prev_x = x_values[i - 1] if i > 0 else x_values[0]
        
        # Calculate change in x normalized to viewport
        change_in_x = ((x_values[i] - prev_x) / range_of_x) * viewport_scale
        
        # Adjust smoothing weight based on x-axis distance
        smoothing_weight_adj = smoothing_weight ** change_in_x
        
        # Update exponential moving average with debias correction
        last_y = last_y * smoothing_weight_adj + y_values[i]
        debias_weight = debias_weight * smoothing_weight_adj + 1
        
        smoothed[i] = last_y / debias_weight
    
    return smoothed


def plot_rewards_from_csv():
    """Read CSV file and plot rewards for 3 seeds."""
    
    # Find CSV file in current directory
    script_dir = Path(__file__).parent
    csv_files = list(script_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV file found in directory!")
        return
    
    csv_path = csv_files[0]
    print(f"Reading: {csv_path.name}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Filter to first 100 episodes
    df = df[df['episode/episode_number'] <= 100]
    print(f"Filtered to {len(df)} episodes (≤100)")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors for each seed
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    seed_names = ['Seed 1', 'Seed 2', 'Seed 3']

    # Smoothing parameter (0.99 is a good default, similar to TensorBoard)
    smoothing_param = 0.99
    
    # Plot each seed
    for col, color, name in zip(['1', '2', '3'], colors, seed_names):
        # Get non-NaN values
        mask = df[col].notna()
        episodes = df.loc[mask, 'episode/episode_number'].values.astype(float)
        rewards = df.loc[mask, col].values.astype(float)
        
        # Plot raw curve in background (transparent)
        ax.plot(episodes, rewards, color=color, linewidth=1.0, alpha=0.2)
        
        # Apply advanced smoothing
        smoothed = advanced_smooth(episodes, rewards, smoothing_param=smoothing_param)
        
        # Plot smoothed line on top
        ax.plot(episodes, smoothed, color=color, label=name, 
                linewidth=2.0, alpha=0.9)
        
        print(f"\n{name}:")
        print(f"  Episodes: {len(rewards)}")
        print(f"  Episode range: [{episodes.min():.0f}, {episodes.max():.0f}]")
        print(f"  Reward range: [{rewards.min():.2f}, {rewards.max():.2f}]")
        print(f"  Mean reward: {rewards.mean():.2f}")
        print(f"  Final smoothed value: {smoothed[-1]:.2f}")
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)  # Set x-axis limit to 100
    
    plt.tight_layout()
    
    # Save as PDF
    output_path = script_dir / "reward_comparison.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"\n✓ Saved plot to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    plot_rewards_from_csv()