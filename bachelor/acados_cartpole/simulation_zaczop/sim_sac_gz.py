"""Gros & Zanon-style MPC-RL in simulation: LEARN the reference, SAMPLE the disturbance.

This is the paper-faithful counterpart to sim_sac_zop.py. The split follows
Gros & Zanon, "Towards Safe Reinforcement Learning Using NMPC and Policy
Gradients: Part I - Stochastic case" (arXiv:1906.04057), Sec. IV-C:

    theta (LEARNED)   the pole-angle reference xref1 of the MPC cost, exactly
                      the parametrization the "full" planner learns.
    d     (SAMPLED)   the gradient disturbance d^T u_0 on the stage-0 cost,
                      Eq. (45), drawn from N(0, sigma_d^2) every step and never
                      optimized. It is the ONLY source of exploration.

Because d enters the constrained NLP rather than being added to the action, every
perturbed action still satisfies the input and state constraints by construction
-- the property the paper is after, and what additive action noise would destroy.

How this differs from the neighbours in this folder:

    sim_sac_zop.py --planner full   learns xref1, explores by SAMPLING xref1
                                    (noise on the learned parameter itself)
    sim_sac_zop.py --planner du0    learns d_u0, references frozen
                                    (inverse of the paper)
    sim_sac_gz.py                   learns xref1 deterministically, explores
                                    via sampled d_u0                (the paper)

Algorithm. Since the actor is DETERMINISTIC and exploration is external, this is
a TD3/DDPG-style deterministic policy gradient rather than SAC -- there is no
entropy term and no log-probability to evaluate (computing the density of the
MPC-induced action distribution would need the Jacobian of Eq. (48), which is
exactly the cost the paper spends its Section V on).

    behaviour   theta = actor(s),  d ~ N(0, sigma_d),  F = MPC(s, [theta, d])
    buffer      (s, [theta, d], r, s', done)   <- the noise IS part of the
                                                  stored action, so the critic
                                                  sees on-policy targets
    critic      Q(s, [theta, d]), twin critics, min over the pair
    target      y = r + gamma * (1-done) * min Q'(s', [theta'(s'), 0])
    actor       maximize min Q(s, [theta(s), 0])   (greedy action has d = 0)

Everything else -- env, reward registry, buffer, network sizes, lr, gamma, tau,
UTD -- is kept identical to sim_sac_zop.py so the comparison stays clean.

    python sim_sac_gz.py --reward cos_bonus_spin3 --seed 0 --steps 10000
"""

import os
import sys
import time
from argparse import ArgumentParser
from collections import deque
from pathlib import Path

# make 'bachelor' importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn as nn
import wandb

from bachelor.acados_cartpole.real_sac_zop.my_sac import SacCritic, SacTrainerConfig
from bachelor.acados_cartpole.real_sac_zop.my_utils import soft_target_update
from bachelor.acados_cartpole.real_sac_zop.my_buffer import ReplayBuffer
from bachelor.acados_cartpole.simulation_zaczop.planner_registry import make_planner
from bachelor.acados_cartpole.simulation_zaczop.sim_env import (
    RealCartPoleSimEnv,
    RealCartPoleSimConfig,
)
from bachelor.acados_cartpole.simulation_zaczop.rewards import get_reward_fn, REWARDS

from leap_c.planner import ControllerFromPlanner
from leap_c.torch.nn.extractor import get_extractor_cls
from leap_c.torch.nn.mlp import Mlp
from leap_c.torch.nn.bounded_distributions import BoundedTransform

import gymnasium as gym

# same key as in real_sac_zop.py; respects an already-set env var / wandb login
os.environ.setdefault('WANDB_API_KEY', 'fd053eb0471b83f999819cd4c4e4930ea28de0ea')

# stabilization tracking (same as sim_sac_zop.py; 200 steps at 10 ms = 2 s)
STABILIZATION_BUFFER_SIZE = 200
STABILIZATION_THRESHOLD = 0.15  # rad


def parse_args():
    p = ArgumentParser(description="Gros & Zanon-style MPC-RL: learned reference, sampled d_u0")
    p.add_argument("--reward", type=str, default="cos_bonus_spin3", choices=sorted(REWARDS),
                   help="Reward function from rewards.py")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=10_000, help="Total env steps")
    p.add_argument("--max-ep-steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.01, help="Sim time step [s]")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--torch-threads", type=int, default=0,
                   help="torch.set_num_threads; 0 = leave default. Use 1-2 for parallel sweeps.")
    # exploration — the whole point of this script
    p.add_argument("--explore-sigma-d", type=float, default=50.0,
                   help="Std of the sampled gradient disturbance d ~ N(0, sigma^2), in d_u0 "
                        "units. Measured du_0/dd is ~-0.19 upright, so sigma=50 moves the applied "
                        "force by roughly 5-9 N there. NOT r*Fmax=2, which would be invisible.")
    p.add_argument("--d-max", type=float, default=300.0,
                   help="Half-width of the d_u0 box. Only has to clear the sampled noise; keep "
                        "it a few sigma above --explore-sigma-d.")
    p.add_argument("--explore-sigma-final", type=float, default=None,
                   help="If set, linearly anneal the disturbance std from --explore-sigma-d to "
                        "this value over training (classic DDPG noise decay). None = constant.")
    # training cadence — identical defaults to sim_sac_zop.py
    p.add_argument("--train-start", type=int, default=1000,
                   help="Env steps before gradient updates begin")
    p.add_argument("--updates-per-step", type=int, default=20,
                   help="Gradient steps per env step")
    p.add_argument("--actor-update-freq", type=int, default=20,
                   help="Actor (and target) update every N gradient steps")
    # logging / output
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default="cartpole-planner-ablation-sim")
    p.add_argument("--wandb-group", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--log-step-freq", type=int, default=200)
    p.add_argument("--learn-log-freq", type=int, default=200)
    p.add_argument("--ckpt-every", type=int, default=50, help="Save checkpoints every N episodes")
    p.add_argument("--runs-dir", type=Path, default=Path(__file__).parent / "runs_gz")
    return p.parse_args()


class DeterministicRefActor(nn.Module):
    """Deterministic actor: obs -> theta (the learned MPC reference).

    No distribution, no sampling — exploration is the MPC's d_u0 channel, not
    noise on this output. The tanh transform keeps theta inside its box.
    """

    def __init__(self, extractor_cls, observation_space, theta_space, mlp_cfg):
        super().__init__()
        self.extractor = extractor_cls(observation_space)
        self.transform = BoundedTransform(theta_space)
        self.mlp = Mlp(
            input_sizes=self.extractor.output_size,
            output_sizes=[theta_space.shape[0]],
            mlp_cfg=mlp_cfg,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        e = self.extractor(obs)
        out = self.mlp(e)
        pre = out[0] if isinstance(out, (tuple, list)) else out
        return self.transform(pre)


def main():
    args = parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = args.device
    run_name = args.run_name or f"gz_{args.reward}_s{args.seed}_{int(time.time())}"
    run_dir = args.runs_dir / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Environment ##############################################################
    sim_cfg = RealCartPoleSimConfig(dt=args.dt)
    reward_fn = get_reward_fn(args.reward)
    env = RealCartPoleSimEnv(reward_fn, cfg=sim_cfg, max_episode_steps=args.max_ep_steps)

    # MPC layer ################################################################
    planner, cfg_planner = make_planner("gz", run_dir / "acados_code", d_max=args.d_max)
    controller_wrapped = ControllerFromPlanner(planner)

    _x_thr = planner.cfg.x_threshold
    obs_low = np.array([-_x_thr, -np.pi, -5, -21], dtype=np.float32)
    obs_high = np.array([_x_thr, np.pi, 5, 21], dtype=np.float32)
    obs_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    # param_space is [xref1, d_u0]; the actor only owns the first entry.
    param_space = controller_wrapped.param_space
    assert int(np.prod(param_space.shape)) == 2, (
        f"gz planner must expose [xref1, d_u0], got shape {param_space.shape}")
    theta_space = gym.spaces.Box(
        low=param_space.low[:1].astype(np.float32),
        high=param_space.high[:1].astype(np.float32),
        dtype=np.float32,
    )
    d_lo, d_hi = float(param_space.low[1]), float(param_space.high[1])

    # Networks — same sizes / hyperparameters as sim_sac_zop.py ################
    cfg = SacTrainerConfig()
    cfg.critic_mlp.norm_layer = "layer_norm"

    replay_buffer = ReplayBuffer(buffer_limit=cfg.buffer_size, device=device)
    extractor_cls = get_extractor_cls("identity")

    # critic sees the FULL param vector [theta, d] — the noise is part of the action
    critic = SacCritic(
        extractor_cls=extractor_cls,
        observation_space=obs_space,
        action_space=param_space,
        mlp_cfg=cfg.critic_mlp,
        num_critics=cfg.num_critics,
    ).to(device)
    target_critic = SacCritic(
        extractor_cls=extractor_cls,
        observation_space=obs_space,
        action_space=param_space,
        mlp_cfg=cfg.critic_mlp,
        num_critics=cfg.num_critics,
    ).to(device)
    target_critic.load_state_dict(critic.state_dict())

    actor = DeterministicRefActor(extractor_cls, obs_space, theta_space, cfg.actor_mlp).to(device)
    target_actor = DeterministicRefActor(
        extractor_cls, obs_space, theta_space, cfg.actor_mlp).to(device)
    target_actor.load_state_dict(actor.state_dict())

    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.lr_q)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr_pi)

    zeros_d = torch.zeros((1, 1), dtype=torch.float32, device=device)

    def greedy_param(obs_batch, net):
        """[theta(s), 0] — the unperturbed (evaluation) parameter vector."""
        theta = net(obs_batch)
        return torch.cat([theta, zeros_d.expand(theta.shape[0], 1)], dim=1)

    # wandb ####################################################################
    wandb.init(
        project=args.wandb_project, name=run_name, group=args.wandb_group,
        mode=args.wandb_mode, dir=str(run_dir),
        config={
            "algo": "gz_ddpg", "planner": "gz", "reward": args.reward, "seed": args.seed,
            "steps": args.steps, "dt": args.dt, "max_ep_steps": args.max_ep_steps,
            "explore_sigma_d": args.explore_sigma_d,
            "explore_sigma_final": args.explore_sigma_final,
            "d_max": args.d_max,
            "train_start": args.train_start, "updates_per_step": args.updates_per_step,
            "actor_update_freq": args.actor_update_freq,
            "buffer_size": cfg.buffer_size, "batch_size": cfg.batch_size,
            "lr_q": cfg.lr_q, "lr_pi": cfg.lr_pi, "gamma": cfg.gamma, "tau": cfg.tau,
            "num_critics": cfg.num_critics,
            "N_horizon": cfg_planner.N_horizon, "T_horizon": cfg_planner.T_horizon,
        },
    )

    def save_checkpoints(episode_count, abs_step_count, total_training_steps, episode_rewards):
        torch.save(critic.state_dict(), ckpt_dir / 'critic.pth')
        torch.save(target_critic.state_dict(), ckpt_dir / 'target_critic.pth')
        torch.save(actor.state_dict(), ckpt_dir / 'actor.pth')
        torch.save(target_actor.state_dict(), ckpt_dir / 'target_actor.pth')
        torch.save({'episode_count': episode_count, 'abs_step_count': abs_step_count,
                    'total_training_steps': total_training_steps,
                    'episode_rewards': episode_rewards}, ckpt_dir / 'meta.pth')
        print(f'Saved checkpoints to {ckpt_dir}')

    # Updates ##################################################################
    state_train = {'total_training_steps': 0}
    metrics = {'q_losses': [], 'pi_losses': [], 'q_values': [], 'q_targets': []}

    def single_update(update_actor: bool):
        if len(replay_buffer) < cfg.batch_size:
            return False

        o, a, r, o_prime, te = replay_buffer.sample(cfg.batch_size)

        with torch.no_grad():
            # target uses the GREEDY next action (d = 0): the disturbance is
            # behaviour-only, it must not leak into the bootstrap target.
            q_target = target_critic(o_prime, greedy_param(o_prime, target_actor))
            q_target = torch.min(q_target, dim=1, keepdim=True).values
            target = r[:, None].to(device) + cfg.gamma * (1 - te[:, None].to(device)) * q_target

        q = critic(o, a)
        q_loss = torch.mean((q - target).pow(2))
        critic_optimizer.zero_grad()
        q_loss.backward()
        critic_optimizer.step()

        if update_actor:
            # deterministic policy gradient: push theta(s) towards higher Q at d = 0
            q_pi = critic(o, greedy_param(o, actor))
            pi_loss = -torch.min(q_pi, dim=1, keepdim=True).values.mean()
            actor_optimizer.zero_grad()
            pi_loss.backward()
            actor_optimizer.step()
            metrics['pi_losses'].append(pi_loss.item())

        if state_train['total_training_steps'] % cfg.soft_update_freq == 0:
            soft_target_update(critic, target_critic, cfg.tau)
            soft_target_update(actor, target_actor, cfg.tau)

        state_train['total_training_steps'] += 1
        metrics['q_losses'].append(q_loss.item())
        metrics['q_values'].append(q.mean().item())
        metrics['q_targets'].append(target.mean().item())
        return True

    def flush_learn_metrics(abs_step_count):
        if not metrics['q_losses']:
            return
        wandb.log({
            'learning/total_training_steps': state_train['total_training_steps'],
            'learning/q_loss_avg': np.mean(metrics['q_losses']),
            'learning/q_avg': np.mean(metrics['q_values']),
            'learning/q_target_avg': np.mean(metrics['q_targets']),
            'learning/pi_loss': np.mean(metrics['pi_losses']) if metrics['pi_losses'] else float('nan'),
        }, step=abs_step_count)
        for v in metrics.values():
            v.clear()

    # Main loop ################################################################
    abs_step_count = 0
    episode_count = 0
    episode_rewards = []
    grad_steps_since_log = 0
    num_terminations = 0
    num_stabilized_steps = 0
    log_every = max(1, args.log_step_freq)
    t_start = time.perf_counter()

    print("=" * 64)
    print(f"Run: {run_name} | reward: {args.reward} | seed: {args.seed}")
    print(f"Steps: {args.steps} | dt: {args.dt}s | planner: gz (learn xref1, sample d_u0)")
    print(f"Exploration: d ~ N(0, {args.explore_sigma_d}^2)"
          + (f" -> {args.explore_sigma_final} (annealed)" if args.explore_sigma_final else "")
          + f" | d box +-{args.d_max}")
    print("=" * 64)

    try:
        while abs_step_count < args.steps:
            obs, _ = env.reset(seed=args.seed + episode_count)
            episode_count += 1
            episode_step_count = 0
            current_episode_reward = 0.0
            max_force_perep = 0.0
            theta_buffer = deque(maxlen=STABILIZATION_BUFFER_SIZE)
            stabilized_this_episode = False
            stabilized_at_step = None

            # warm-start the solver on the first state, like sim_sac_zop.py
            obs_batch = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                ctx_current, _, _, _, _ = planner(obs_batch, ctx=None)

            done = False
            while not done and abs_step_count < args.steps:
                episode_step_count += 1
                abs_step_count += 1

                obs_batch = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

                # --- theta: deterministic, learned -------------------------
                with torch.no_grad():
                    theta = actor(obs_batch)

                # --- d: sampled, never learned (Gros & Zanon Eq. (45)) ------
                if args.explore_sigma_final is not None:
                    frac = min(1.0, abs_step_count / max(1, args.steps))
                    sigma_d = (args.explore_sigma_d
                               + frac * (args.explore_sigma_final - args.explore_sigma_d))
                else:
                    sigma_d = args.explore_sigma_d
                d_sample = float(np.clip(rng.normal(0.0, sigma_d), d_lo, d_hi))
                d_t = torch.full((1, 1), d_sample, dtype=torch.float32, device=device)

                param_applied = torch.cat([theta, d_t], dim=1)

                with torch.no_grad():
                    ctx_current, action = controller_wrapped(
                        obs_batch, param_applied, ctx=ctx_current)
                u_force = float(action[0].cpu().numpy().squeeze())
                max_force_perep = max(max_force_perep, abs(u_force))

                obs_next, reward, terminated, truncated, info = env.step(u_force)
                done = terminated or truncated
                if terminated:
                    num_terminations += 1
                current_episode_reward += reward

                theta_buffer.append(float(obs_next[1]))
                if not stabilized_this_episode and len(theta_buffer) == STABILIZATION_BUFFER_SIZE:
                    if all(abs(t) <= STABILIZATION_THRESHOLD for t in theta_buffer):
                        stabilized_this_episode = True
                        stabilized_at_step = episode_step_count
                        num_stabilized_steps += STABILIZATION_BUFFER_SIZE
                elif stabilized_this_episode:
                    num_stabilized_steps += 1

                # the stored action is the FULL [theta, d] actually applied
                replay_buffer.put((
                    torch.as_tensor(obs, dtype=torch.float32),
                    param_applied[0].detach().cpu().float(),
                    float(reward),
                    torch.as_tensor(obs_next, dtype=torch.float32),
                    int(terminated),
                ))

                if abs_step_count >= args.train_start:
                    for _ in range(args.updates_per_step):
                        upd_actor = (
                            state_train['total_training_steps'] % args.actor_update_freq == 0)
                        if single_update(upd_actor):
                            grad_steps_since_log += 1
                    if grad_steps_since_log >= args.learn_log_freq:
                        flush_learn_metrics(abs_step_count)
                        grad_steps_since_log = 0

                if args.log_step_freq and abs_step_count % log_every == 0:
                    wandb.log({
                        'step/u_force': u_force,
                        'step/theta_ref': float(theta[0, 0].cpu()),
                        'step/d_sample': d_sample,
                        'step/sigma_d': sigma_d,
                        'step/x': float(obs_next[0]),
                        'step/theta': float(obs_next[1]),
                        'step/v': float(obs_next[2]),
                        'step/thetadot': float(obs_next[3]),
                        'step/reward': reward,
                        'step/episode_step': episode_step_count,
                        'terminations/total': num_terminations,
                        'step/stabilized': int(stabilized_this_episode),
                        'stabilization/stabilized_steps_total': num_stabilized_steps,
                    }, step=abs_step_count)

                obs = obs_next

            episode_rewards.append({'episode': episode_count,
                                    'cumulative_reward': current_episode_reward,
                                    'steps': episode_step_count})
            elapsed = time.perf_counter() - t_start
            sps = abs_step_count / elapsed
            print(f"Ep {episode_count}: reward = {current_episode_reward:.2f}, "
                  f"steps = {episode_step_count}, max |F| = {max_force_perep:.1f} N, "
                  f"stabilized = {stabilized_this_episode}, "
                  f"total steps = {abs_step_count} ({sps:.0f} steps/s)")

            ep_log = {
                'episode/episode_number': episode_count,
                'episode/cumulative_reward': current_episode_reward,
                'episode/steps': episode_step_count,
                'episode/max_force': max_force_perep,
                'episode/stabilized': int(stabilized_this_episode),
                'episode/steps_per_second': sps,
                'episode/terminated': int(terminated),
                'terminations/total': num_terminations,
            }
            if stabilized_at_step is not None:
                ep_log['stabilization/achieved_at_step'] = stabilized_at_step
            wandb.log(ep_log, step=abs_step_count)

            if episode_count % args.ckpt_every == 0:
                save_checkpoints(episode_count, abs_step_count,
                                 state_train['total_training_steps'], episode_rewards)

    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoints...")

    finally:
        flush_learn_metrics(abs_step_count)
        save_checkpoints(episode_count, abs_step_count,
                         state_train['total_training_steps'], episode_rewards)
        elapsed = time.perf_counter() - t_start
        print(f"\nDone: {abs_step_count} env steps, "
              f"{state_train['total_training_steps']} gradient steps "
              f"in {elapsed/60:.1f} min ({abs_step_count/max(elapsed,1e-9):.0f} steps/s)")
        wandb.finish()


if __name__ == "__main__":
    main()
