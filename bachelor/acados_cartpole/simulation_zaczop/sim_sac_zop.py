"""SAC-ZOP training in simulation, mirroring real_sac_zop.py without the Arduino side.

Same actor/critic/buffer/update code and hyperparameters as the hardware script,
but the environment is RealCartPoleSimEnv (matched to the real setup) and
everything runs synchronously in one process — no serial threads, no semaphores.

Made for quickly testing different reward functions (see rewards.py):

    python sim_sac_zop.py --reward default --seed 0 --steps 200000
    python sim_sac_zop.py --reward energy --run-name energy_test --wandb-mode online

Each run gets its own directory under runs/<run_name>/ with checkpoints,
acados generated code and (optionally) policy heatmaps. For parallel sweeps
over rewards/seeds use run_parallel.py.
"""

import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

# make 'bachelor' importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import wandb
from collections import deque

from bachelor.acados_cartpole.real_sac_zop.my_sac import SacCritic
from bachelor.acados_cartpole.real_sac_zop.my_sac_zop import MpcSacActor, SacZopTrainerConfig
from bachelor.acados_cartpole.real_sac_zop.my_utils import soft_target_update
from bachelor.acados_cartpole.real_sac_zop.my_buffer import ReplayBuffer
from bachelor.acados_cartpole.simulation_zaczop.planner_registry import (
    build_planner,
    PLANNER_REGISTRY,
)
from bachelor.acados_cartpole.my_utils_plot import plot_policy_heatmap
from bachelor.acados_cartpole.simulation_zaczop.sim_env import (
    RealCartPoleSimEnv,
    RealCartPoleSimConfig,
)
from bachelor.acados_cartpole.simulation_zaczop.rewards import get_reward_fn, REWARDS

from leap_c.planner import ControllerFromPlanner
from leap_c.torch.nn.extractor import get_extractor_cls

import gymnasium as gym

# same key as in real_sac_zop.py; respects an already-set env var / wandb login
os.environ.setdefault('WANDB_API_KEY', 'fd053eb0471b83f999819cd4c4e4930ea28de0ea')

# stabilization tracking (same as real script; 200 steps at 10 ms = 2 s)
STABILIZATION_BUFFER_SIZE = 200
STABILIZATION_THRESHOLD = 0.15  # rad


def str2bool(v):
    """Parse a --flag true/false string into a bool."""
    return str(v).strip().lower() in ("true", "1", "yes", "t", "y")


def parse_args():
    p = ArgumentParser(description="SAC-ZOP simulation training (reward function testing)")
    p.add_argument("--reward", type=str, default="cos_bonus_spin", choices=sorted(REWARDS),
                   help="Reward function from rewards.py")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--steps", type=int, default=200_000, help="Total env steps")
    p.add_argument("--max-ep-steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.01, help="Sim time step [s] (real control freq: 10 ms)")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--planner", type=str, default="full", choices=sorted(PLANNER_REGISTRY),
                   help="Which planner/OCP to use (see planner_registry.py)")
    p.add_argument("--torch-threads", type=int, default=0,
                   help="torch.set_num_threads; 0 = leave default. Use 1-2 for parallel sweeps.")
    # training cadence
    p.add_argument("--train-start", type=int, default=1000,
                   help="Env steps before gradient updates begin")
    p.add_argument("--updates-per-step", type=int, default=20,
                   help="Gradient steps per env step")
    p.add_argument("--actor-update-freq", type=int, default=20,
                   help="Actor update every N gradient steps (paper: 1 per 20)")
    # K-step (action-repeat) pattern from real_sac_zop.py
    p.add_argument("--K_step", type=str2bool, default=False,
                   help="Enable the K-step pattern: the actor picks a param every K env steps, "
                        "the MPC re-solves each step with the HELD param, and one "
                        "accumulated-reward transition is stored per cycle. true/false.")
    p.add_argument("--K", type=int, default=5,
                   help="Env steps per RL decision when --K_step true (real script uses 5)")
    # logging / output
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default="cartpole-planner-ablation-sim")
    p.add_argument("--wandb-group", type=str, default=None,
                   help="Group runs in wandb (e.g. one group per sweep)")
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--log-step-freq", type=int, default=200,
                   help="Log step-level stats every N env steps (0 = off)")
    p.add_argument("--learn-log-freq", type=int, default=200,
                   help="Log averaged learning metrics every N gradient steps")
    p.add_argument("--heatmap-every", type=int, default=0,
                   help="Generate policy heatmaps every N episodes (0 = off)")
    p.add_argument("--ckpt-every", type=int, default=50, help="Save checkpoints every N episodes")
    p.add_argument("--runs-dir", type=Path, default=Path(__file__).parent / "runs")
    return p.parse_args()


def main():
    args = parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    # seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    run_name = args.run_name or f"sim_{args.reward}_s{args.seed}_{int(time.time())}"
    run_dir = args.runs_dir / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Environment ##############################################################
    sim_cfg = RealCartPoleSimConfig(dt=args.dt)
    reward_fn = get_reward_fn(args.reward)
    env = RealCartPoleSimEnv(reward_fn, cfg=sim_cfg, max_episode_steps=args.max_ep_steps)

    # MPC layer — the planner (and thus the OCP / parameter interface) is chosen
    # from the planner registry via --planner. The env always passes the full
    # [x, theta, v, thetadot] state, so each planner adapts internally.
    planner = build_planner(args.planner, run_dir / "acados_code")
    controller_wrapped = ControllerFromPlanner(planner)

    # observation / action spaces — identical to real_sac_zop.py
    _x_thr = planner.cfg.x_threshold
    obs_low = np.array([-_x_thr, -np.pi, -5, -21], dtype=np.float32)
    obs_high = np.array([_x_thr, np.pi, 5, 21], dtype=np.float32)
    obs_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
    action_space = controller_wrapped.param_space

    # SAC-ZOP setup — identical to real_sac_zop.py ############################
    cfg_saczop = SacZopTrainerConfig()
    cfg_saczop.critic_mlp.norm_layer = "layer_norm"

    replay_buffer = ReplayBuffer(buffer_limit=cfg_saczop.buffer_size, device=device)
    extractor_cls = get_extractor_cls("identity")

    critic = SacCritic(
        extractor_cls=extractor_cls,
        observation_space=obs_space,
        action_space=action_space,
        mlp_cfg=cfg_saczop.critic_mlp,
        num_critics=cfg_saczop.num_critics,
    ).to(device)

    target_critic = SacCritic(
        extractor_cls=extractor_cls,
        observation_space=obs_space,
        action_space=action_space,
        mlp_cfg=cfg_saczop.critic_mlp,
        num_critics=cfg_saczop.num_critics,
    ).to(device)
    target_critic.load_state_dict(critic.state_dict())

    actor = MpcSacActor(
        extractor_cls=extractor_cls,
        observation_space=obs_space,
        controller=controller_wrapped,
        distribution_name=cfg_saczop.distribution_name,
        mlp_cfg=cfg_saczop.actor_mlp,
        init_param_with_default=cfg_saczop.init_param_with_default,
    ).to(device)

    log_alpha = torch.nn.Parameter(
        torch.tensor(cfg_saczop.init_alpha, dtype=torch.float32, device=device).log()
    )
    alpha_optimizer = (
        torch.optim.Adam([log_alpha], lr=cfg_saczop.lr_alpha)
        if cfg_saczop.lr_alpha is not None
        else None
    )

    param_dim = int(np.prod(action_space.shape))
    action_dim = 1
    entropy_norm = param_dim / action_dim
    # match pure SAC (sim_sac.py): target entropy = -action_dim (= -1.0), not the
    # config's -2.0, so both baselines use the same temperature target.
    target_entropy = -float(action_dim)

    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg_saczop.lr_q)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg_saczop.lr_pi)

    # wandb ###################################################################
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        group=args.wandb_group,
        mode=args.wandb_mode,
        dir=str(run_dir),
        config={
            "planner": args.planner,
            "reward": args.reward,
            "seed": args.seed,
            "steps": args.steps,
            "dt": args.dt,
            "max_ep_steps": args.max_ep_steps,
            "train_start": args.train_start,
            "updates_per_step": args.updates_per_step,
            "actor_update_freq": args.actor_update_freq,
            "K_step": args.K_step,
            "K": args.K,
            "buffer_size": cfg_saczop.buffer_size,
            "batch_size": cfg_saczop.batch_size,
            "lr_q": cfg_saczop.lr_q,
            "lr_pi": cfg_saczop.lr_pi,
            "lr_alpha": cfg_saczop.lr_alpha,
            "gamma": cfg_saczop.gamma,
            "tau": cfg_saczop.tau,
            "num_critics": cfg_saczop.num_critics,
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

    # training step (same math as saczop_single_step_update in real_sac_zop.py)
    state_train = {'total_training_steps': 0}
    metrics = {'q_losses': [], 'pi_losses': [], 'alphas': [],
               'q_values': [], 'q_targets': [], 'entropies': []}

    def single_update(update_actor: bool):
        if len(replay_buffer) < cfg_saczop.batch_size:
            return False

        o, a, r, o_prime, te = replay_buffer.sample(cfg_saczop.batch_size)
        alpha = log_alpha.exp().item()

        with torch.no_grad():
            pi_o_prime = actor(o_prime, None, only_param=True)
            q_target = target_critic(o_prime, pi_o_prime.param)
            q_target = torch.min(q_target, dim=1, keepdim=True).values
            factor = cfg_saczop.entropy_reward_bonus / entropy_norm
            q_target = q_target - alpha * pi_o_prime.log_prob * factor
            target = r[:, None].to(device) + cfg_saczop.gamma * (1 - te[:, None].to(device)) * q_target

        if update_actor:
            pi_o = actor(o, None, only_param=True)
            a_pi = pi_o.param
            log_p = pi_o.log_prob / entropy_norm

            if alpha_optimizer is not None:
                alpha_loss = -torch.mean(log_alpha.exp() * (log_p + target_entropy).detach())
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

        q = critic(o, a)
        q_loss = torch.mean((q - target).pow(2))
        critic_optimizer.zero_grad()
        q_loss.backward()
        critic_optimizer.step()

        if update_actor:
            q_pi = critic(o, a_pi)
            min_q_pi = torch.min(q_pi, dim=1, keepdim=True).values
            pi_loss = (alpha * log_p - min_q_pi).mean()
            actor_optimizer.zero_grad()
            pi_loss.backward()
            actor_optimizer.step()

        if state_train['total_training_steps'] % cfg_saczop.soft_update_freq == 0:
            soft_target_update(critic, target_critic, cfg_saczop.tau)

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
    rl_step_count = 0  # RL decisions (== env steps in standard mode, every K-th in K-step mode)
    episode_rewards = []
    grad_steps_since_log = 0
    num_terminations = 0  # cumulative count of episodes that ended in a trip (terminated)
    num_stabilized_steps = 0  # cumulative count of steps spent in the balanced/stabilized mode
    # log step-level stats every rl_log_every RL steps (keeps the env-step density
    # of log_step_freq the same whether or not K-step is on)
    rl_log_every = max(1, args.log_step_freq // args.K) if args.K_step else max(1, args.log_step_freq)
    t_start = time.perf_counter()

    print("=" * 60)
    print(f"Run: {run_name} | reward: {args.reward} | seed: {args.seed}")
    print(f"Steps: {args.steps} | dt: {args.dt}s"
          + (f" | K-step ON (K={args.K}, actor every {args.K} env steps)" if args.K_step else ""))
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

            # initialize MPC solver with the first state (same as real script)
            obs_batch = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                ctx_planner, _, _, _, _ = planner(obs_batch, ctx=None)
                pi_out = actor(obs_batch, ctx_planner, deterministic=False)
            ctx_current = pi_out.ctx

            # K-step cycle state (only used when args.K_step): the actor picks a
            # param every K env steps, the MPC re-solves each step with the HELD
            # param, and one accumulated-reward transition is stored per cycle.
            step_in_cycle = 0
            accumulated_reward = 0.0
            obs_start_cycle = None
            param_current = None

            done = False
            while not done and abs_step_count < args.steps:
                episode_step_count += 1
                abs_step_count += 1

                obs_batch = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

                if not args.K_step:
                    # standard: call the actor every env step
                    with torch.no_grad():
                        pi_out = actor(obs_batch, ctx_current, deterministic=False)
                    param_current = pi_out.param[0].detach()
                    ctx_current = pi_out.ctx
                    u_force = float(pi_out.action[0].cpu().numpy().squeeze())
                elif step_in_cycle == 0:
                    # K-step, cycle start: call the actor for a fresh param
                    obs_start_cycle = torch.as_tensor(obs, dtype=torch.float32)
                    accumulated_reward = 0.0
                    with torch.no_grad():
                        pi_out = actor(obs_batch, ctx_current, deterministic=False)
                    param_current = pi_out.param[0].detach()
                    ctx_current = pi_out.ctx
                    u_force = float(pi_out.action[0].cpu().numpy().squeeze())
                else:
                    # K-step, intermediate: hold the param, MPC re-solves this state
                    with torch.no_grad():
                        ctx_current, action = controller_wrapped(
                            obs_batch, param_current.unsqueeze(0), ctx=ctx_current)
                    u_force = float(action[0].cpu().numpy().squeeze())

                max_force_perep = max(max_force_perep, abs(u_force))

                obs_next, reward, terminated, truncated, info = env.step(u_force)
                done = terminated or truncated
                if terminated:
                    num_terminations += 1  # this step tripped the episode (|x| > x_threshold)
                current_episode_reward += reward

                # stabilization tracking (per env step, same as real script)
                theta_buffer.append(float(obs_next[1]))
                if not stabilized_this_episode and len(theta_buffer) == STABILIZATION_BUFFER_SIZE:
                    if all(abs(t) <= STABILIZATION_THRESHOLD for t in theta_buffer):
                        stabilized_this_episode = True
                        stabilized_at_step = episode_step_count
                        num_stabilized_steps += STABILIZATION_BUFFER_SIZE  # retroactively credit the 200-step balanced window
                elif stabilized_this_episode:
                    num_stabilized_steps += 1  # latched: every step after achievement counts as stabilized

                obs_next_t = torch.as_tensor(obs_next, dtype=torch.float32)

                # store transition — per env step (standard) or one accumulated-reward
                # transition per K-step cycle (cycle-start obs), like real_sac_zop.py
                if not args.K_step:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32)
                    replay_buffer.put((obs_t, param_current.cpu().float(),
                                       float(reward), obs_next_t, int(terminated)))
                    is_rl_step = True
                    rl_reward = reward
                else:
                    accumulated_reward += reward
                    step_in_cycle += 1
                    is_rl_step = (step_in_cycle == args.K or done)
                    if is_rl_step:
                        replay_buffer.put((obs_start_cycle, param_current.cpu().float(),
                                           float(accumulated_reward), obs_next_t, int(terminated)))
                        step_in_cycle = 0
                    rl_reward = accumulated_reward

                # everything below runs ONCE per RL step (every K-th env step in
                # K-step mode): gradient updates, learning + step-level logging
                if is_rl_step:
                    rl_step_count += 1

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

                    if args.log_step_freq and rl_step_count % rl_log_every == 0:
                        wandb.log({
                            'step/u_force': u_force,
                            'step/param': param_current.cpu().numpy().tolist()
                            if param_current.dim() > 0 else param_current.item(),
                            'step/x': float(obs_next[0]),
                            'step/theta': float(obs_next[1]),
                            'step/v': float(obs_next[2]),
                            'step/thetadot': float(obs_next[3]),
                            'step/reward': rl_reward,
                            'step/episode_step': episode_step_count,
                            'step/rl_step': rl_step_count,
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

            # periodic checkpoints / heatmaps
            if episode_count % args.ckpt_every == 0:
                save_checkpoints(episode_count, abs_step_count,
                                 state_train['total_training_steps'], episode_rewards)
            if args.heatmap_every and episode_count % args.heatmap_every == 0:
                try:
                    save_checkpoints(episode_count, abs_step_count,
                                     state_train['total_training_steps'], episode_rewards)
                    plot_policy_heatmap(
                        actor_path=str(ckpt_dir / 'actor.pth'),
                        v_fixed=0.0,
                        thetadot_fixed=0.0,
                        x_range=(-0.35, 0.35),
                        theta_range=(-np.pi, np.pi),
                        resolution=50,
                        plt_show=False,
                        episode_num=episode_count,
                    )
                except Exception as e:
                    print(f"Failed to generate policy heatmaps: {e}")

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
