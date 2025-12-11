"""Convert simulation checkpoints to real hardware format.

This script converts checkpoints from the Leap-C SacZopTrainer format
to the format used by real_sac_zop.py for hardware deployment.

Usage:
    python scripts/convert_sim_to_real.py --input output/2025_12_11/.../ckpts --output bachelor/acados_cartpole/real_sac_zop/checkpoints
"""

import argparse
from pathlib import Path
import torch


def convert_checkpoints(input_dir: Path, output_dir: Path, ckpt_mode: str = "best"):
    """Convert simulation checkpoints to real hardware format.
    
    Args:
        input_dir: Path to the simulation ckpts directory (e.g., output/.../ckpts/)
        output_dir: Path to the real hardware checkpoints directory
        ckpt_mode: Which checkpoints to load ('best' or 'last')
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapping from simulation to real hardware names
    mapping = {
        f"{ckpt_mode}_pi.ckpt": "actor.pth",
        f"{ckpt_mode}_q.ckpt": "critic.pth",
        f"{ckpt_mode}_q_target.ckpt": "target_critic.pth",
        "last_log_alpha.ckpt": "log_alpha.pth",  # log_alpha is always 'last'
    }
    
    print(f"Converting checkpoints from simulation to real hardware format...")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()
    
    for sim_name, real_name in mapping.items():
        sim_path = input_dir / sim_name
        real_path = output_dir / real_name
        
        if not sim_path.exists():
            print(f"⚠️  Warning: {sim_name} not found, skipping...")
            continue
        
        # Load and re-save (this ensures format compatibility)
        checkpoint = torch.load(sim_path, map_location="cpu", weights_only=False)
        torch.save(checkpoint, real_path)
        
        print(f"✓ Converted {sim_name} → {real_name}")
    
    # Also load trainer_state to extract meta information
    trainer_state_path = input_dir / f"{ckpt_mode}_trainer_state.ckpt"
    if trainer_state_path.exists():
        trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
        
        # Create meta.pth with default values (you can adjust these)
        meta = {
            'episode_count': 0,  # Reset for real hardware
            'learning_step': 0,  # Reset for real hardware
            'abs_step_count': 0,  # Reset for real hardware
            'episode_rewards': [],
            'wandb_run_id': None,  # Will be set on first real hardware run
            'pretrained_from_sim': True,
            'sim_training_steps': trainer_state.step if hasattr(trainer_state, 'step') else 0,
        }
        
        torch.save(meta, output_dir / "meta.pth")
        print(f"✓ Created meta.pth with pretrained flag")
    
    print()
    print("✅ Conversion complete!")
    print(f"You can now use these checkpoints in real_sac_zop.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert sim checkpoints to real hardware format")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to simulation ckpts directory (e.g., output/2025_12_11/.../ckpts/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bachelor/acados_cartpole/real_sac_zop/checkpoints"),
        help="Path to real hardware checkpoints directory (default: bachelor/acados_cartpole/real_sac_zop/checkpoints)",
    )
    parser.add_argument(
        "--ckpt-mode",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Which checkpoint to convert (default: best)",
    )
    
    args = parser.parse_args()
    convert_checkpoints(args.input, args.output, args.ckpt_mode)
