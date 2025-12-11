"""Pretrain SAC-ZOP in simulation with custom cartpole controller.

This script trains SAC-ZOP in the CartPole simulation environment using
the custom planner defined in bachelor/acados_cartpole/my_planner.py.

The trained model can then be transferred to real hardware using
convert_sim_to_real.py.

Usage:
    python scripts/pretrain_sac_zop.py --seed 0 --train-steps 200000
"""

from argparse import ArgumentParser
from dataclasses import asdict, field
from pathlib import Path

from bachelor.acados_cartpole.my_planner import (
    CartPolePlannerConfig,
    CartPolePlanner,
    create_custom_cartpole_params,
)
from bachelor.acados_cartpole.my_env import CartPoleEnv
from leap_c.run import default_name, default_output_path, init_run
from leap_c.torch.nn.extractor import get_extractor_cls
from leap_c.torch.rl.sac_zop import SacZopTrainer, SacZopTrainerConfig


def create_custom_controller(reuse_code_dir: Path | None = None):
    """Create the custom CartPole planner."""
    cfg_planner = CartPolePlannerConfig()
    params = create_custom_cartpole_params("stagewise", cfg_planner.N_horizon)
    
    # Reuse compiled code if directory is provided
    export_dir = reuse_code_dir / "cartpole" if reuse_code_dir else None
    
    from leap_c.planner import ControllerFromPlanner
    planner = CartPolePlanner(cfg_planner, params, export_directory=export_dir)
    return ControllerFromPlanner(planner)


def create_custom_env():
    """Create the custom CartPole environment."""
    return CartPoleEnv()


def create_pretrain_config(seed: int, train_steps: int) -> SacZopTrainerConfig:
    """Create configuration for pretraining."""
    cfg = SacZopTrainerConfig()
    
    # Training settings
    cfg.seed = seed
    cfg.train_steps = train_steps
    cfg.train_start = 5000  # Start training after collecting some data
    cfg.val_freq = 10_000
    cfg.val_num_rollouts = 20
    cfg.val_deterministic = True
    cfg.val_num_render_rollouts = 0
    cfg.val_render_mode = "rgb_array"
    cfg.val_report_score = "cum"
    cfg.ckpt_modus = "best"
    
    # SAC-ZOP hyperparameters
    cfg.batch_size = 64
    cfg.buffer_size = 1_000_000
    cfg.gamma = 0.99
    cfg.tau = 0.005
    cfg.soft_update_freq = 1
    cfg.lr_q = 0.001
    cfg.lr_pi = 0.001
    cfg.lr_alpha = 0.001
    cfg.init_alpha = 0.02
    cfg.target_entropy = None
    cfg.entropy_reward_bonus = True
    cfg.num_critics = 2
    cfg.update_freq = 4
    cfg.distribution_name = "squashed_gaussian"
    cfg.init_param_with_default = True
    
    # Network architecture
    cfg.critic_mlp.hidden_dims = (256, 256, 256)
    cfg.critic_mlp.activation = "relu"
    cfg.critic_mlp.weight_init = "orthogonal"
    
    cfg.actor_mlp.hidden_dims = (256, 256, 256)
    cfg.actor_mlp.activation = "relu"
    cfg.actor_mlp.weight_init = "orthogonal"
    
    # Logging
    cfg.log.verbose = True
    cfg.log.interval = 1_000
    cfg.log.window = 10_000
    cfg.log.csv_logger = True
    cfg.log.tensorboard_logger = True
    cfg.log.wandb_logger = False
    cfg.log.wandb_init_kwargs = {}
    
    return cfg


def pretrain_sac_zop(
    cfg: SacZopTrainerConfig,
    output_path: Path,
    device: str = "cpu",
    reuse_code_dir: Path | None = None,
) -> float:
    """Run the SAC-ZOP pretraining.
    
    Args:
        cfg: The training configuration.
        output_path: Path to save outputs and checkpoints.
        device: The device to use ('cpu' or 'cuda').
        reuse_code_dir: Directory to reuse compiled controller code from.
        
    Returns:
        The final validation score.
    """
    print("=" * 80)
    print("SAC-ZOP Pretraining for Real Hardware")
    print("=" * 80)
    print()
    
    trainer = SacZopTrainer(
        cfg=cfg,
        train_env=create_custom_env(),
        val_env=create_custom_env(),
        controller=create_custom_controller(reuse_code_dir),
        extractor_cls=get_extractor_cls("identity"),
        output_path=output_path,
        device=device,
    )
    
    # Initialize run (saves config, loads checkpoints if continuing)
    init_run(trainer, cfg, output_path)
    
    # Run training
    final_score = trainer.run()
    
    print()
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Final validation score: {final_score:.2f}")
    print(f"Checkpoints saved to: {output_path / 'ckpts'}")
    print()
    print("To convert for real hardware, run:")
    print(f"  python scripts/convert_sim_to_real.py --input {output_path / 'ckpts'}")
    print()
    
    return final_score


if __name__ == "__main__":
    parser = ArgumentParser(description="Pretrain SAC-ZOP in simulation for real hardware")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--train-steps", 
        type=int, 
        default=200_000, 
        help="Number of training steps"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Custom output path (default: auto-generated)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--reuse-code",
        action="store_true",
        help="Reuse compiled controller code (faster startup)",
    )
    parser.add_argument(
        "--reuse-code-dir",
        type=Path,
        default=None,
        help="Directory with compiled code to reuse",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (username or team)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="cartpole-pretrain",
        help="W&B project name",
    )
    
    args = parser.parse_args()
    
    # Create configuration
    cfg = create_pretrain_config(args.seed, args.train_steps)
    
    # Configure W&B if requested
    if args.use_wandb:
        cfg.log.wandb_logger = True
        cfg.log.wandb_init_kwargs = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "name": default_name(args.seed, tags=["pretrain", "cartpole"]),
            "config": asdict(cfg),
        }
    
    # Determine output path
    if args.output_path is None:
        output_path = default_output_path(
            seed=args.seed, 
            tags=["pretrain_sac_zop", "cartpole"]
        )
    else:
        output_path = args.output_path
    
    # Determine code reuse directory
    if args.reuse_code and args.reuse_code_dir is None:
        from leap_c.run import default_controller_code_path
        reuse_code_dir = default_controller_code_path()
    elif args.reuse_code_dir is not None:
        reuse_code_dir = args.reuse_code_dir
    else:
        reuse_code_dir = None
    
    # Run pretraining
    pretrain_sac_zop(
        cfg=cfg,
        output_path=output_path,
        device=args.device,
        reuse_code_dir=reuse_code_dir,
    )
