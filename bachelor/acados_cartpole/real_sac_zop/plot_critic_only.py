"""
Standalone script to generate critic Q-value heatmaps from checkpoint.
Looks for critic.pth and actor.pth in the checkpoints directory.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import gymnasium as gym

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from bachelor.acados_cartpole.my_utils_plot import plot_policy_heatmap
from bachelor.acados_cartpole.real_sac_zop.my_sac import SacCritic
from bachelor.acados_cartpole.real_sac_zop.my_sac_zop import MpcSacActor, SacZopTrainerConfig
from bachelor.acados_cartpole.my_planner import (
    CartPolePlannerConfig,
    CartPolePlanner,
    create_custom_cartpole_params
)
from leap_c.planner import ControllerFromPlanner
from leap_c.torch.nn.extractor import get_extractor_cls

def main():
    # Define checkpoint directory
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    actor_path = os.path.join(checkpoint_dir, "actor.pth")
    critic_path = os.path.join(checkpoint_dir, "critic.pth")
    
    # Check if checkpoints exist
    if not os.path.exists(actor_path):
        print(f"Error: Actor checkpoint not found at {actor_path}")
        return
    
    if not os.path.exists(critic_path):
        print(f"Warning: Critic checkpoint not found at {critic_path}")
        print("Will generate plots without critic Q-values")
        critic_path = None
    
    print(f"Found actor checkpoint: {actor_path}")
    if critic_path:
        print(f"Found critic checkpoint: {critic_path}")
    
    # ========== Initialize all the same components as real_sac_zop.py ==========
    
    # Device setup
    device = "cpu"
    
    # MPC Layer Setup
    cfg_planner = CartPolePlannerConfig()
    params = create_custom_cartpole_params("global", cfg_planner.N_horizon)
    planner = CartPolePlanner(cfg_planner, params)
    controller_wrapped = ControllerFromPlanner(planner)
    
    # Observation and action spaces
    _x_thr = getattr(cfg_planner, "x_threshold", 0.4)
    _x_low = -float(_x_thr)
    _x_high = float(_x_thr)
    
    obs_low = np.array([_x_low, -np.pi, -5, -21], dtype=np.float32)
    obs_high = np.array([_x_high, np.pi, 5, 21], dtype=np.float32)
    obs_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
    action_space = controller_wrapped.param_space
    
    # SacZop config
    cfg_saczop = SacZopTrainerConfig()
    cfg_saczop.critic_mlp.norm_layer = "layer_norm"
    
    # Extractor
    extractor_cls = get_extractor_cls("identity")
    
    # Initialize critic
    critic = SacCritic(
        extractor_cls=extractor_cls,
        observation_space=obs_space,
        action_space=action_space,
        mlp_cfg=cfg_saczop.critic_mlp,
        num_critics=cfg_saczop.num_critics,
    ).to(device)
    
    # Initialize actor
    actor = MpcSacActor(
        extractor_cls=extractor_cls,
        observation_space=obs_space,
        controller=controller_wrapped,
        distribution_name=cfg_saczop.distribution_name,
        mlp_cfg=cfg_saczop.actor_mlp,
        init_param_with_default=cfg_saczop.init_param_with_default,
    ).to(device)
    
    # Load checkpoints
    print("\nLoading model checkpoints...")
    try:
        actor.load_state_dict(torch.load(actor_path, map_location=device))
        print(f"✓ Loaded actor from {actor_path}")
    except Exception as e:
        print(f"✗ Failed to load actor: {e}")
        return
    
    if critic_path:
        try:
            critic.load_state_dict(torch.load(critic_path, map_location=device))
            print(f"✓ Loaded critic from {critic_path}")
        except Exception as e:
            print(f"✗ Failed to load critic: {e}")
            critic = None
    else:
        critic = None
    
    # Set models to eval mode
    actor.eval()
    if critic:
        critic.eval()
    
    print("\nGenerating heatmap grid (3x3)...")
    print("-" * 60)
    
    # Generate heatmap grid using the existing function
    try:
        plot_policy_heatmap(
            actor_path=actor_path,
            critic_path=critic_path if critic else None,
            x_range=(-0.35, 0.35),
            theta_range=(-3.14159, 3.14159),
            resolution=50,
            plt_show=False,
            save_path=None,
            episode_num=None
        )
    except Exception as e:
        print(f"Error generating heatmaps: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("-" * 60)
    print(f"✓ Heatmaps saved to: {checkpoint_dir}")
    print("\nGenerated files:")
    print("  - policy_heatmap_theta_ref_grid.pdf")
    print("  - policy_heatmap_mpc_force_grid.pdf")
    if critic:
        print("  - policy_heatmap_critic_grid.pdf")
    print("  - policy_heatmap_force_grid.pdf")

if __name__ == "__main__":
    main()