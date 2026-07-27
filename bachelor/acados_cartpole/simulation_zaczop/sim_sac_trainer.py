"""Pure SAC baseline using the canonical SacTrainer from my_sac.py (1:1).

Instead of a hand-written loop (see sim_sac.py), this wires RealCartPoleSimEnv
into the leap_c-style Trainer framework and simply calls ``SacTrainer.run()``.
The learning code is exactly ``SacTrainer.train_loop`` in my_sac.py — standard
SAC: one gradient step per env step (update_freq=1), actor updated every step,
``num_critics`` Q-networks, automatic temperature tuning. No MPC layer: the
policy outputs the cart force directly.

    python sim_sac_trainer.py --reward default --seed 0 --steps 200000

For a 5-seed sweep into a fresh wandb project use:
    python run_parallel_sac.py --script sim_sac_trainer.py --wandb-project cartpole-sac-trainer-sim
"""

import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

# make 'bachelor' importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from bachelor.acados_cartpole.real_sac_zop.my_sac import SacTrainer, SacTrainerConfig
from bachelor.acados_cartpole.simulation_zaczop.sim_env import (
    RealCartPoleSimEnv,
    RealCartPoleSimConfig,
)
from bachelor.acados_cartpole.simulation_zaczop.rewards import get_reward_fn, REWARDS
from leap_c.utils.logger import LoggerConfig

# same key as the rest of the project; respects an already-set env var / wandb login
os.environ.setdefault('WANDB_API_KEY', 'fd053eb0471b83f999819cd4c4e4930ea28de0ea')


def parse_args():
    p = ArgumentParser(description="Pure SAC baseline via the canonical SacTrainer (my_sac.py)")
    p.add_argument("--reward", type=str, default="default", choices=sorted(REWARDS),
                   help="Reward function from rewards.py")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200_000, help="Total env steps (train_steps)")
    p.add_argument("--train-start", type=int, default=1000,
                   help="Env steps before gradient updates begin")
    p.add_argument("--max-ep-steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.01, help="Sim time step [s] (real control freq: 10 ms)")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--torch-threads", type=int, default=0,
                   help="torch.set_num_threads; 0 = leave default. Use 1-2 for parallel sweeps.")
    p.add_argument("--critic-layer-norm", action="store_true", default=True,
                   help="Use layer_norm in the critic MLP (matches the saczop setup)")
    p.add_argument("--no-critic-layer-norm", dest="critic_layer_norm", action="store_false")
    # validation cadence (Trainer framework)
    p.add_argument("--val-freq", type=int, default=10_000, help="Run validation every N steps")
    p.add_argument("--val-rollouts", type=int, default=5, help="Episodes per validation")
    # logging / output
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default="cartpole-sac-trainer-sim")
    p.add_argument("--wandb-group", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--runs-dir", type=Path, default=Path(__file__).parent / "runs_sac_trainer")
    return p.parse_args()


def make_env(reward: str, dt: float, max_ep_steps: int) -> RealCartPoleSimEnv:
    cfg = RealCartPoleSimConfig(dt=dt)
    return RealCartPoleSimEnv(get_reward_fn(reward), cfg=cfg, max_episode_steps=max_ep_steps)


def main():
    args = parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    run_name = args.run_name or f"sactr_{args.reward}_s{args.seed}_{int(time.time())}"
    run_dir = args.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # separate envs for training and validation (validation acts deterministically)
    train_env = make_env(args.reward, args.dt, args.max_ep_steps)
    val_env = make_env(args.reward, args.dt, args.max_ep_steps)

    # logging: Logger calls wandb.init(**wandb_init_kwargs)
    log_cfg = LoggerConfig(
        wandb_logger=(args.wandb_mode != "disabled"),
        wandb_init_kwargs=dict(
            project=args.wandb_project,
            name=run_name,
            group=args.wandb_group,
            mode=args.wandb_mode,
            config={
                "algo": "sac_trainer",
                "reward": args.reward,
                "seed": args.seed,
                "steps": args.steps,
                "dt": args.dt,
                "max_ep_steps": args.max_ep_steps,
                "critic_layer_norm": args.critic_layer_norm,
            },
        ),
        tensorboard_logger=False,
        csv_logger=True,
    )

    cfg = SacTrainerConfig(
        seed=args.seed,
        train_steps=args.steps,
        train_start=args.train_start,
        val_freq=args.val_freq,
        val_num_rollouts=args.val_rollouts,
        val_deterministic=True,
        val_num_render_rollouts=0,   # the sim env cannot render
        val_render_mode=None,
        val_report_score="cum",
        ckpt_modus="best",
        log=log_cfg,
    )
    if args.critic_layer_norm:
        cfg.critic_mlp.norm_layer = "layer_norm"

    print("=" * 60)
    print(f"Run: {run_name} | reward: {args.reward} | seed: {args.seed} | algo: SacTrainer (my_sac)")
    print(f"Steps: {args.steps} | dt: {args.dt}s | layer_norm: {args.critic_layer_norm}")
    print("=" * 60)

    trainer = SacTrainer(
        cfg=cfg,
        val_env=val_env,
        output_path=run_dir,
        device=args.device,
        train_env=train_env,
        extractor_cls="identity",
    )

    score = trainer.run()
    print(f"\nDone. Reported validation score ({cfg.val_report_score}): {score:.3f}")


if __name__ == "__main__":
    main()
