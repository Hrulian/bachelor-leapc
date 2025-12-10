import torch
import matplotlib.pyplot as plt
import os

# Path to checkpoints directory
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
META_PATH = os.path.join(CHECKPOINT_DIR, "meta.pth")

def plot_rewards():
    """Load meta.pth and plot episode rewards."""
    
    if not os.path.exists(META_PATH):
        print(f"Error: {META_PATH} not found")
        return
    
    # Load meta data
    meta = torch.load(META_PATH, map_location='cpu')
    episode_rewards = meta.get('episode_rewards', [])
    
    if not episode_rewards:
        print("No episode rewards found in meta.pth")
        return
    
    # Extract episode numbers and cumulative rewards
    episodes = [ep_data['episode'] for ep_data in episode_rewards]
    rewards = [ep_data['cumulative_reward'] for ep_data in episode_rewards]
    steps = [ep_data['steps'] for ep_data in episode_rewards]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot cumulative reward per episode
    ax1.plot(episodes, rewards, marker='o', linestyle='-', markersize=3)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('Episode Reward over Training')
    ax1.grid(True, alpha=0.3)
    
    # Plot steps per episode
    ax2.plot(episodes, steps, marker='o', linestyle='-', markersize=3, color='orange')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Steps per Episode')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(CHECKPOINT_DIR, "reward_plot.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    
    # Show plot
    plt.show()
    
    # Print statistics
    print(f"\nTraining Statistics:")
    print(f"Total episodes: {len(episodes)}")
    print(f"Average reward: {sum(rewards)/len(rewards):.2f}")
    print(f"Max reward: {max(rewards):.2f} (Episode {episodes[rewards.index(max(rewards))]})")
    print(f"Min reward: {min(rewards):.2f} (Episode {episodes[rewards.index(min(rewards))]})")
    print(f"Average steps: {sum(steps)/len(steps):.1f}")

if __name__ == "__main__":
    plot_rewards()
