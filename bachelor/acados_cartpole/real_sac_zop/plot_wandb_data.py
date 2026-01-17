import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration
WANDB_PROJECT = "cartpole-sac-zop-1"
WANDB_ENTITY = None  # Set to your wandb username if needed, otherwise None
RUN_ID = None  # Set to specific run ID, or None to use latest run

# Metrics to plot
METRICS = {
    'step/reward': {
        'title': 'Step Reward over Time',
        'ylabel': 'Reward',
        'xlabel': 'Absolute Step',
        'smooth_window': 100,
    },
    'episode/cumulative_reward': {
        'title': 'Episode Cumulative Reward',
        'ylabel': 'Cumulative Reward',
        'xlabel': 'Episode',
        'smooth_window': 10,
    },
    'learning/q_loss': {
        'title': 'Critic Loss over Training',
        'ylabel': 'Q Loss',
        'xlabel': 'Absolute Step',
        'smooth_window': 100,
    },
    'learning/pi_loss': {
        'title': 'Actor Loss over Training',
        'ylabel': 'Policy Loss',
        'xlabel': 'Absolute Step',
        'smooth_window': 100,
    },
    'learning/alpha': {
        'title': 'Entropy Temperature (α) over Training',
        'ylabel': 'Alpha',
        'xlabel': 'Absolute Step',
        'smooth_window': 100,
    },
    'learning/entropy': {
        'title': 'Policy Entropy over Training',
        'ylabel': 'Entropy',
        'xlabel': 'Absolute Step',
        'smooth_window': 100,
    },
}


def smooth_data(data, window_size):
    """Apply moving average smoothing to data."""
    if len(data) < window_size:
        return data
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().values


def get_run(project, entity=None, run_id=None):
    """Get a wandb run by ID or return the latest run."""
    api = wandb.Api()
    
    if run_id:
        # Get specific run by ID
        if entity:
            run_path = f"{entity}/{project}/{run_id}"
        else:
            run_path = f"{project}/{run_id}"
        run = api.run(run_path)
        print(f"Loaded run: {run.name} (ID: {run.id})")
    else:
        # Get latest run
        runs = api.runs(f"{entity}/{project}" if entity else project)
        if not runs:
            raise ValueError(f"No runs found in project {project}")
        run = runs[0]  # Most recent run
        print(f"Loaded latest run: {run.name} (ID: {run.id})")
    
    return run


def fetch_metric_data(run, metric_name):
    """Fetch time series data for a specific metric."""
    history = run.scan_history(keys=[metric_name, '_step'])
    
    steps = []
    values = []
    
    for row in history:
        if metric_name in row and row[metric_name] is not None:
            steps.append(row.get('_step', len(steps)))
            values.append(row[metric_name])
    
    return np.array(steps), np.array(values)


def plot_metric(ax, steps, values, config):
    """Plot a single metric with smoothing."""
    # Plot raw data
    ax.plot(steps, values, alpha=0.3, label='Raw', linewidth=0.5)
    
    # Plot smoothed data
    if len(values) > config['smooth_window']:
        smoothed = smooth_data(values, config['smooth_window'])
        ax.plot(steps, smoothed, label=f"Smoothed (window={config['smooth_window']})", linewidth=2)
    
    ax.set_title(config['title'], fontsize=14, fontweight='bold')
    ax.set_xlabel(config['xlabel'], fontsize=12)
    ax.set_ylabel(config['ylabel'], fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_multiple_runs(project, entity=None, run_ids=None, metric='step/reward'):
    """Compare multiple runs for a single metric."""
    api = wandb.Api()
    
    if run_ids is None:
        # Get last 5 runs
        runs = api.runs(f"{entity}/{project}" if entity else project, per_page=5)
        run_ids = [r.id for r in runs]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for run_id in run_ids:
        if entity:
            run_path = f"{entity}/{project}/{run_id}"
        else:
            run_path = f"{project}/{run_id}"
        
        run = api.run(run_path)
        steps, values = fetch_metric_data(run, metric)
        
        if len(values) > 0:
            # Smooth and plot
            if len(values) > 100:
                smoothed = smooth_data(values, 100)
                ax.plot(steps, smoothed, label=f"{run.name}", linewidth=2, alpha=0.8)
            else:
                ax.plot(steps, values, label=f"{run.name}", linewidth=2, alpha=0.8)
    
    ax.set_title(f'Comparison: {metric}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel(metric.split('/')[-1].replace('_', ' ').title(), fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main plotting function."""
    # Initialize wandb API
    api = wandb.Api()
    
    # Get run
    run = get_run(WANDB_PROJECT, WANDB_ENTITY, RUN_ID)
    
    print(f"\nRun configuration:")
    print(f"  Created: {run.created_at}")
    print(f"  State: {run.state}")
    print(f"  Duration: {run.summary.get('_runtime', 'N/A')} seconds")
    
    # Create output directory
    output_dir = Path(__file__).parent / "wandb_plots"
    output_dir.mkdir(exist_ok=True)
    print(f"\nSaving plots to: {output_dir}")
    
    # Plot each metric
    for metric_name, config in METRICS.items():
        print(f"\nFetching {metric_name}...")
        steps, values = fetch_metric_data(run, metric_name)
        
        if len(values) == 0:
            print(f"  No data found for {metric_name}")
            continue
        
        print(f"  Found {len(values)} data points")
        print(f"  Range: [{values.min():.3f}, {values.max():.3f}]")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_metric(ax, steps, values, config)
        
        # Save plot
        safe_name = metric_name.replace('/', '_')
        output_path = output_dir / f"{safe_name}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
    
    # Create combined plot for main metrics
    print("\nCreating combined plot...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Training Overview: {run.name}', fontsize=16, fontweight='bold')
    
    main_metrics = [
        'step/reward',
        'episode/cumulative_reward',
        'learning/alpha',
        'learning/entropy',
    ]
    
    for ax, metric_name in zip(axes.flat, main_metrics):
        if metric_name in METRICS:
            steps, values = fetch_metric_data(run, metric_name)
            if len(values) > 0:
                plot_metric(ax, steps, values, METRICS[metric_name])
    
    plt.tight_layout()
    combined_path = output_dir / "combined_overview.png"
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined plot: {combined_path}")
    plt.close()
    
    # Optional: Show plots
    # plt.show()
    
    print("\n✓ Done! All plots saved.")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Episode rewards
    _, ep_rewards = fetch_metric_data(run, 'episode/cumulative_reward')
    if len(ep_rewards) > 0:
        print(f"\nEpisode Rewards:")
        print(f"  Total episodes: {len(ep_rewards)}")
        print(f"  Mean: {ep_rewards.mean():.2f}")
        print(f"  Std: {ep_rewards.std():.2f}")
        print(f"  Min: {ep_rewards.min():.2f}")
        print(f"  Max: {ep_rewards.max():.2f}")
        if len(ep_rewards) >= 10:
            print(f"  Last 10 mean: {ep_rewards[-10:].mean():.2f}")
    
    # Learning metrics
    _, alpha_values = fetch_metric_data(run, 'learning/alpha')
    if len(alpha_values) > 0:
        print(f"\nEntropy Temperature (α):")
        print(f"  Initial: {alpha_values[0]:.4f}")
        print(f"  Final: {alpha_values[-1]:.4f}")
        print(f"  Mean: {alpha_values.mean():.4f}")


if __name__ == "__main__":
    main()