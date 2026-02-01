import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import FuncFormatter

def thousands_formatter(x, pos):
    """Format y-axis labels to use 'k' for thousands."""
    if x >= 1000:
        return f'{x/1000:.0f}k'
    else:
        return f'{x:.0f}'

def plot_cumulative_from_csv():
    """Read CSV file, compute cumulative sum for each seed, and plot."""
    
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
    
    # Get episode numbers (first column)
    episodes = df['episode/episode_number'].values
    
    # Get data columns (all except first)
    data_columns = [col for col in df.columns if col != 'episode/episode_number']
    
    print(f"\nData columns: {data_columns}")
    print(f"Episodes: {len(episodes)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors for each seed
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    seed_names = ['Seed 1', 'Seed 2', 'Seed 3']
    
    # Process each seed separately
    for i, (col, color, name) in enumerate(zip(data_columns, colors, seed_names)):
        # Get values for this seed
        values = df[col].values.astype(float)
        
        # Compute cumulative sum for this seed
        cumulative = np.cumsum(values)
        
        print(f"\n{name} (column '{col}'):")
        print(f"  First 3 values: {values[:3]}")
        print(f"  First 3 cumulative: {cumulative[:3]}")
        print(f"  Final cumulative: {cumulative[-1]:.0f}")
        
        # Plot
        ax.plot(episodes, cumulative, color=color, label=name, 
                linewidth=2.0, alpha=0.9)
    
    # Configure plot
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Learning Steps', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(episodes[0], episodes[-1])
    
    # Format y-axis to use 'k' notation
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    
    plt.tight_layout()
    
    # Save as PDF
    output_path = script_dir / "cumulative_plot.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"\n✓ Saved plot to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    plot_cumulative_from_csv()