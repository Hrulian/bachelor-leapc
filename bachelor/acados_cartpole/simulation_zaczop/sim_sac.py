"""Pure SAC training in simulation (no MPC layer) — baseline for the SAC-ZOP runs.

Same environment (RealCartPoleSimEnv), same replay buffer, same reward functions
and the SAME networks as the hardware/ZOP code, but the policy is a plain
``SacActor`` that outputs the cart force directly (1-D action in [-Fmax, Fmax]).
No acados planner, no controller context, no parameter-space exploration — this
is the vanilla SAC baseline to compare the MPC+RL approach against.

    python sim_sac.py --reward default --seed 0 --steps 200000

Each run gets its own directory under runs/<run_name>/ with checkpoints.
For a 5-seed sweep into a fresh wandb project use run_parallel_sac.py.
"""

import os
import sys
import time
from argparse import ArgumentParser
from collections import deque
from pathlib import Path

# make 'bachelor' importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import gymnasium as gym
import numpy as np
import torch
import wandb

from bachelor.acados_cartpole.real_sac_zop.my_sac import (
    SacActor,
    SacCritic,
    SacTrainerConfig,
)
from bachelor.acados_cartpole.real_sac_zop.my_utils import soft_target_update
from bachelor.acados_cartpole.real_sac_zop.my_buffer import ReplayBuffer
from bachelor.acados_cartpole.simulation_zaczop.sim_env import (
    RealCartPoleSimEnv,
    RealCartPoleSimConfig,
)
from bachelor.acados_cartpole.simulation_zaczop.rewards import get_reward_fn, REWARDS

from leap_c.torch.nn.extractor import get_extractor_cls

# same key as the rest of the project; respects an already-set env var / wandb login
os.environ.setdefault('WANDB_API_KEY', 'fd053eb0471b83f999819cd4c4e4930ea28de0ea')

# stabilization tracking (same as the ZOP script; 200 steps at 10 ms = 2 s)
STABILIZATION_BUFFER_SIZE = 200
STABILIZATION_THRESHOLD = 0.15  # rad


def parse_args():
    p = ArgumentParser(description="Pure SAC simulation training (MPC-free baseline)")
    p.add_argument("--reward", type=str, default="default", choices=sorted(REWARDS),
                   help="Reward function from rewards.py")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200_000, help="Total env steps")
    p.add_argument("--max-ep-steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.01, help="Sim time step [s] (real control freq: 10 ms)")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--torch-threads", type=int, default=0,
                   help="torch.set_num_threads; 0 = leave default. Use 1-2 for parallel sweeps.")
    # training cadence
    p.add_argument("--train-start", type=int, default=1000,
                   help="Env steps before gradient updates begin")
    p.add_argument("--updates-per-step", type=int, default=20,
                   help="Gradient (critic) steps per env step — UTD ratio (saczop: 20)")
    p.add_argument("--actor-update-freq", type=int, default=20,
                   help="Actor update every N gradient steps (saczop: 1 per 20)")
    # logging / output
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default="cartpole-sac-pure-sim")
    p.add_argument("--wandb-group", type=str, default=None,
                   help="Group runs in wandb (e.g. one group per sweep)")
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--log-step-freq", type=int, default=200,
                   help="Log step-level stats every N env steps (0 = off)")
    p.add_argument("--learn-log-freq", type=int, default=200,
                   help="Log averaged learning metrics every N gradient steps")
    p.add_argument("--ckpt-every", type=int, default=50, help="Save checkpoints every N episodes")
    p.add_argument("--runs-dir", type=Path, default=Path(__file__).parent / "runs_sac")
    return p.parse_args()


def main():
    args = parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    # seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    run_name = args.run_name or f"sac_{args.reward}_s{args.seed}_{int(time.time())}"
    run_dir = args.runs_dir / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Environment ##############################################################
    sim_cfg = RealCartPoleSimConfig(dt=args.dt)
    reward_fn = get_reward_fn(args.reward)
    env = RealCartPoleSimEnv(reward_fn, cfg=sim_cfg, max_episode_steps=args.max_ep_steps)

    # action = cart force directly (1-D), bounded by Fmax — no MPC in between
    Fmax = float(sim_cfg.Fmax)
    action_space = gym.spaces.Box(low=-Fmax, high=Fmax, shape=(1,), dtype=np.float32)
    obs_space = env.observation_space
    action_dim = action_space.shape[0]

    # SAC setup — same networks/hyperparameters as the hardware code ###########
    cfg_sac = SacTrainerConfig()
    cfg_sac.critic_mlp.norm_layer = "layer_norm"  # match sim_sac_zop.py

    replay_buffer = ReplayBuffer(buffer_limit=cfg_sac.buffer_size, device=device)
    extractor_cls = get_extractor_cls("identity")

    critic = SacCritic(
        extractor_cls=extractor_cls,
        action_space=action_space,
        observation_space=obs_space,
        mlp_cfg=cfg_sac.critic_mlp,
        num_critics=cfg_sac.num_critics,
    ).to(device)

    target_critic = SacCritic(
        extractor_cls=extractor_cls,
        action_space=action_space,
        observation_space=obs_space,
        mlp_cfg=cfg_sac.critic_mlp,
        num_critics=cfg_sac.num_critics,
    ).to(device)
    target_critic.load_state_dict(critic.state_dict())

    actor = SacActor(
        extractor_cls=extractor_cls,
        action_space=action_space,
        observation_space=obs_space,
        distribution_name=cfg_sac.distribution_name,
        mlp_cfg=cfg_sac.actor_mlp,
    ).to(device)

    log_alpha = torch.nn.Parameter(
        torch.tensor(cfg_sac.init_alpha, dtype=torch.float32, device=device).log()
    )
    alpha_optimizer = (
        torch.optim.Adam([log_alpha], lr=cfg_sac.lr_alpha)
        if cfg_sac.lr_alpha is not None
        else None
    )
    # standard SAC heuristic for a 1-D continuous action
    target_entropy = -float(action_dim)

    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg_sac.lr_q)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg_sac.lr_pi)

    # wandb ###################################################################
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        group=args.wandb_group,
        mode=args.wandb_mode,
        dir=str(run_dir),
        config={
            "algo": "sac_pure",
            "reward": args.reward,
            "seed": args.seed,
            "steps": args.steps,
            "dt": args.dt,
            "max_ep_steps": args.max_ep_steps,
            "train_start": args.train_start,
            "updates_per_step": args.updates_per_step,
            "actor_update_freq": args.actor_update_freq,
            "Fmax": Fmax,
            "buffer_size": cfg_sac.buffer_size,
            "batch_size": cfg_sac.batch_size,
            "lr_q": cfg_sac.lr_q,
            "lr_pi": cfg_sac.lr_pi,
            "lr_alpha": cfg_sac.lr_alpha,
            "gamma": cfg_sac.gamma,
            "tau": cfg_sac.tau,
            "num_critics": cfg_sac.num_critics,
            "target_entropy": target_entropy,
        },
    )

    # checkpointing ###########################################################
    def ckpt_paths():
        return {
            'critic': ckpt_dir / 'critic.pth',
            'target_critic': ckpt_dir / 'target_critic.pth',
            'actor': ckpt_dir / 'actor.pth',
            'log_alpha': ckpt_dir / 'log_alpha.pth',
            'meta': ckpt_dir / 'meta.pth',
        }

    def save_checkpoints(episode_count, abs_step_count, total_training_steps, episode_rewards):
        paths = ckpt_paths()
        torch.save(critic.state_dict(), paths['critic'])
        torch.save(target_critic.state_dict(), paths['target_critic'])
        torch.save(actor.state_dict(), paths['actor'])
        torch.save(log_alpha.detach().cpu(), paths['log_alpha'])
        torch.save({
            'episode_count': episode_count,
            'abs_step_count': abs_step_count,
            'total_training_steps': total_training_steps,
            'episode_rewards': episode_rewards,
        }, paths['meta'])
        print(f'Saved checkpoints to {ckpt_dir}')

    # training step (vanilla SAC, mirrors SacTrainer.train_loop in my_sac.py) ##
    state_train = {'total_training_steps': 0}
    metrics = {'q_losses': [], 'pi_losses': [], 'alphas': [],
               'q_values': [], 'q_targets': [], 'entropies': []}

    def single_update(update_actor: bool):
        if len(replay_buffer) < cfg_sac.batch_size:
            return False

        o, a, r, o_prime, te = replay_buffer.sample(cfg_sac.batch_size)
        alpha = log_alpha.exp().item()

        # critic target (always)
        with torch.no_grad():
            a_pi_prime, log_p_prime, _ = actor(o_prime)
            q_target = target_critic(o_prime, a_pi_prime)
            q_target = torch.min(q_target, dim=1, keepdim=True).values
            q_target = q_target - alpha * log_p_prime * cfg_sac.entropy_reward_bonus
            target = r[:, None].to(device) + cfg_sac.gamma * (1 - te[:, None].to(device)) * q_target

        # actor forward + temperature update — only on actor steps
        if update_actor:
            a_pi, log_p, _ = actor(o)
            if alpha_optimizer is not None:
                alpha_loss = -torch.mean(log_alpha.exp() * (log_p + target_entropy).detach())
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

        # critic update (always)
        q = critic(o, a)
        q_loss = torch.mean((q - target).pow(2))
        critic_optimizer.zero_grad()
        q_loss.backward()
        critic_optimizer.step()

        # actor update — only on actor steps
        if update_actor:
            q_pi = critic(o, a_pi)
            min_q_pi = torch.min(q_pi, dim=1, keepdim=True).values
            pi_loss = (alpha * log_p - min_q_pi).mean()
            actor_optimizer.zero_grad()
            pi_loss.backward()
            actor_optimizer.step()

        if state_train['total_training_steps'] % cfg_sac.soft_update_freq == 0:
            soft_target_update(critic, target_critic, cfg_sac.tau)

        state_train['total_training_steps'] += 1

        metrics['q_losses'].append(q_loss.item())
        metrics['q_values'].append(q.mean().item())
        metrics['q_targets'].append(target.mean().item())
        if update_actor:
            metrics['pi_losses'].append(pi_loss.item())
            metrics['alphas'].append(alpha)
            metrics['entropies'].append(-log_p.mean().item())
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
            'learning/alpha': metrics['alphas'][-1] if metrics['alphas'] else float('nan'),
            'learning/entropy': np.mean(metrics['entropies']) if metrics['entropies'] else float('nan'),
        }, step=abs_step_count)
        for v in metrics.values():
            v.clear()

    # Main RL loop ############################################################
    abs_step_count = 0
    episode_count = 0
    episode_rewards = []
    grad_steps_since_log = 0
    num_terminations = 0  # cumulative count of episodes that ended in a trip (terminated)
    num_stabilized_steps = 0  # cumulative count of steps spent in the balanced/stabilized mode
    t_start = time.perf_counter()

    print("=" * 60)
    print(f"Run: {run_name} | reward: {args.reward} | seed: {args.seed} | algo: pure SAC")
    print(f"Steps: {args.steps} | dt: {args.dt}s")
    print("=" * 60)

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

            done = False
            while not done and abs_step_count < args.steps:
                episode_step_count += 1
                abs_step_count += 1

                obs_batch = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action, _, _ = actor(obs_batch, deterministic=False)
                a_np = action[0].cpu().numpy()           # shape (1,)
                u_force = float(a_np.squeeze())
                max_force_perep = max(max_force_perep, abs(u_force))

                obs_next, reward, terminated, truncated, info = env.step(a_np)
                done = terminated or truncated
                if terminated:
                    num_terminations += 1  # this step tripped the episode (|x| > x_threshold)
                current_episode_reward += reward

                # stabilization tracking (same as ZOP script)
                theta_buffer.append(float(obs_next[1]))
                if not stabilized_this_episode and len(theta_buffer) == STABILIZATION_BUFFER_SIZE:
                    if all(abs(t) <= STABILIZATION_THRESHOLD for t in theta_buffer):
                        stabilized_this_episode = True
                        stabilized_at_step = episode_step_count
                        num_stabilized_steps += STABILIZATION_BUFFER_SIZE  # retroactively credit the 200-step balanced window
                elif stabilized_this_episode:
                    num_stabilized_steps += 1  # latched: every step after achievement counts as stabilized

                # store transition (action = force)
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                obs_next_t = torch.as_tensor(obs_next, dtype=torch.float32)
                action_t = action[0].detach().cpu().float()
                replay_buffer.put((obs_t, action_t, float(reward), obs_next_t, int(terminated)))

                # gradient updates — UTD = updates_per_step critic steps per env step,
                # with one actor step per actor_update_freq (saczop: 20 critic, 1 actor)
                if abs_step_count >= args.train_start:
                    for _ in range(args.updates_per_step):
                        update_actor = (
                            state_train['total_training_steps'] % args.actor_update_freq == 0
                        )
                        if single_update(update_actor):
                            grad_steps_since_log += 1
                    if grad_steps_since_log >= args.learn_log_freq:
                        flush_learn_metrics(abs_step_count)
                        grad_steps_since_log = 0

                # step-level logging
                if args.log_step_freq and abs_step_count % args.log_step_freq == 0:
                    wandb.log({
                        'step/u_force': u_force,
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

            # episode end #####################################################
            episode_rewards.append({
                'episode': episode_count,
                'cumulative_reward': current_episode_reward,
                'steps': episode_step_count,
            })
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
