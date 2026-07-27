"""
Benchmark: one SAC-ZOP training block (20 gradient steps).
Compares eager vs torch.compile for the critic networks.

Run with:
    python -m bachelor.acados_cartpole.real_sac_zop.benchmark_block
"""

import statistics
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from bachelor.acados_cartpole.my_planner import (
    CartPolePlannerConfig,
    CartPolePlanner,
    create_custom_cartpole_params,
)
from bachelor.acados_cartpole.real_sac_zop.my_buffer import ReplayBuffer
from bachelor.acados_cartpole.real_sac_zop.my_sac import SacCritic
from bachelor.acados_cartpole.real_sac_zop.my_sac_zop import MpcSacActor, SacZopTrainerConfig
from bachelor.acados_cartpole.real_sac_zop.my_utils import soft_target_update
from leap_c.planner import ControllerFromPlanner
from leap_c.torch.nn.extractor import get_extractor_cls

# ── tuneable knobs ─────────────────────────────────────────────────────────────
DEVICE          = "cpu"
N_WARMUP        = 5    # blocks before timing (cache warmup / JIT trigger)
N_BENCH         = 50   # blocks to measure
BUFFER_FILL     = 5_000  # transitions pre-loaded into buffer
DROPOUT_P       = 0.1  # dropout prob for the dropout-critic comparison


# ── setup ──────────────────────────────────────────────────────────────────────
def setup():
    torch.manual_seed(42)
    np.random.seed(42)

    cfg_planner = CartPolePlannerConfig()
    planner = CartPolePlanner(cfg_planner, create_custom_cartpole_params("global", cfg_planner.N_horizon))
    ctrl = ControllerFromPlanner(planner)

    x_thr = float(getattr(cfg_planner, "x_threshold", 0.5))
    obs_space = gym.spaces.Box(
        low =np.array([-x_thr, -np.pi, -5.0, -21.0], dtype=np.float32),
        high=np.array([ x_thr,  np.pi,  5.0,  21.0], dtype=np.float32),
    )
    action_space = ctrl.param_space

    cfg = SacZopTrainerConfig()
    cfg.critic_mlp.norm_layer = "layer_norm"

    extractor_cls = get_extractor_cls("identity")

    critic = SacCritic(
        extractor_cls=extractor_cls,
        observation_space=obs_space,
        action_space=action_space,
        mlp_cfg=cfg.critic_mlp,
        num_critics=cfg.num_critics,
    ).to(DEVICE)

    target_critic = SacCritic(
        extractor_cls=extractor_cls,
        observation_space=obs_space,
        action_space=action_space,
        mlp_cfg=cfg.critic_mlp,
        num_critics=cfg.num_critics,
    ).to(DEVICE)
    target_critic.load_state_dict(critic.state_dict())

    actor = MpcSacActor(
        extractor_cls=extractor_cls,
        observation_space=obs_space,
        controller=ctrl,
        distribution_name=cfg.distribution_name,
        mlp_cfg=cfg.actor_mlp,
        init_param_with_default=cfg.init_param_with_default,
    ).to(DEVICE)

    log_alpha = torch.nn.Parameter(
        torch.tensor(cfg.init_alpha, dtype=torch.float32).log()
    ).to(DEVICE)

    param_dim     = int(np.prod(action_space.shape))
    entropy_norm  = float(param_dim)  # action_dim = 1
    target_entropy = cfg.target_entropy if cfg.target_entropy is not None else -1.0

    buf = ReplayBuffer(buffer_limit=cfg.buffer_size, device=DEVICE)
    for _ in range(BUFFER_FILL):
        buf.put((
            torch.tensor(obs_space.sample(),    dtype=torch.float32),
            torch.tensor(action_space.sample(), dtype=torch.float32),
            float(np.random.randn()),
            torch.tensor(obs_space.sample(),    dtype=torch.float32),
            int(np.random.rand() < 0.05),
        ))

    print(f"  obs_dim={obs_space.shape[0]}  param_dim={param_dim}  batch={cfg.batch_size}")
    print(f"  critic: {cfg.num_critics}×MLP{list(cfg.critic_mlp.hidden_dims)} + {cfg.critic_mlp.norm_layer}")
    print(f"  buffer: {len(buf)} transitions")

    return dict(
        cfg           = cfg,
        critic        = critic,
        target_critic = target_critic,
        actor         = actor,
        log_alpha     = log_alpha,
        critic_opt    = torch.optim.Adam(critic.parameters(), lr=cfg.lr_q),
        actor_opt     = torch.optim.Adam(actor.parameters(),  lr=cfg.lr_pi),
        alpha_opt     = torch.optim.Adam([log_alpha],         lr=cfg.lr_alpha),
        buf           = buf,
        entropy_norm  = entropy_norm,
        target_entropy= target_entropy,
        # kept so a dropout variant of the critic can be rebuilt 1:1
        obs_space     = obs_space,
        action_space  = action_space,
        extractor_cls = extractor_cls,
    )


# ── single gradient step ───────────────────────────────────────────────────────
def single_step_old(comp, step_idx: int, actor_update_freq: int = 5):
    """OLD version: else-branches compute actor+q_pi on every non-actor step."""
    cfg = comp['cfg']; critic = comp['critic']; target_critic = comp['target_critic']
    actor = comp['actor']; log_alpha = comp['log_alpha']
    critic_opt = comp['critic_opt']; actor_opt = comp['actor_opt']; alpha_opt = comp['alpha_opt']
    buf = comp['buf']; entropy_norm = comp['entropy_norm']; target_entropy = comp['target_entropy']

    update_actor = (step_idx % actor_update_freq == 0)
    o, a, r, op, te = buf.sample(cfg.batch_size)

    with torch.no_grad():
        pi_op = actor(op, None, only_param=True)
        q_tgt = target_critic(op, pi_op.param)
        q_tgt = torch.min(q_tgt, dim=1, keepdim=True).values
        factor = cfg.entropy_reward_bonus / entropy_norm
        q_tgt = q_tgt - log_alpha.exp().item() * pi_op.log_prob * factor
        target = r[:, None] + cfg.gamma * (1 - te[:, None]) * q_tgt

    if update_actor:
        pi_o = actor(o, None, only_param=True)
        log_p = pi_o.log_prob / entropy_norm
        alpha_loss = -torch.mean(log_alpha.exp() * (log_p + target_entropy).detach())
        alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()
    else:
        with torch.no_grad():
            pi_o = actor(o, None, only_param=True)
            log_p = pi_o.log_prob / entropy_norm

    q = critic(o, a)
    q_loss = torch.mean((q - target).pow(2))
    critic_opt.zero_grad(); q_loss.backward(); critic_opt.step()

    if update_actor:
        q_pi = critic(o, pi_o.param)
        min_q_pi = torch.min(q_pi, dim=1, keepdim=True).values
        pi_loss = (log_alpha.exp().item() * log_p - min_q_pi).mean()
        actor_opt.zero_grad(); pi_loss.backward(); actor_opt.step()
    else:
        with torch.no_grad():
            q_pi = critic(o, pi_o.param)
            min_q_pi = torch.min(q_pi, dim=1, keepdim=True).values
            _ = (log_alpha.exp().item() * log_p - min_q_pi).mean()  # metrics only

    soft_target_update(critic, target_critic, cfg.tau)


def single_step_new(comp, step_idx: int, actor_update_freq: int = 20):
    """NEW version: no else-branches, alpha cached, fewer .item() syncs."""
    cfg = comp['cfg']; critic = comp['critic']; target_critic = comp['target_critic']
    actor = comp['actor']; log_alpha = comp['log_alpha']
    critic_opt = comp['critic_opt']; actor_opt = comp['actor_opt']; alpha_opt = comp['alpha_opt']
    buf = comp['buf']; entropy_norm = comp['entropy_norm']; target_entropy = comp['target_entropy']

    update_actor = (step_idx % actor_update_freq == 0)
    o, a, r, op, te = buf.sample(cfg.batch_size)

    alpha = log_alpha.exp().item()  # cached once

    with torch.no_grad():
        pi_op = actor(op, None, only_param=True)
        q_tgt = target_critic(op, pi_op.param)
        q_tgt = torch.min(q_tgt, dim=1, keepdim=True).values
        factor = cfg.entropy_reward_bonus / entropy_norm
        q_tgt = q_tgt - alpha * pi_op.log_prob * factor
        target = r[:, None] + cfg.gamma * (1 - te[:, None]) * q_tgt

    if update_actor:
        pi_o = actor(o, None, only_param=True)
        log_p = pi_o.log_prob / entropy_norm
        alpha_loss = -torch.mean(log_alpha.exp() * (log_p + target_entropy).detach())
        alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()

    q = critic(o, a)
    q_loss = torch.mean((q - target).pow(2))
    critic_opt.zero_grad(); q_loss.backward(); critic_opt.step()

    if update_actor:
        q_pi = critic(o, pi_o.param)
        min_q_pi = torch.min(q_pi, dim=1, keepdim=True).values
        pi_loss = (alpha * log_p - min_q_pi).mean()
        actor_opt.zero_grad(); pi_loss.backward(); actor_opt.step()

    soft_target_update(critic, target_critic, cfg.tau)


# ── block = 20 steps ───────────────────────────────────────────────────────────
def run_block(comp, step_fn, actor_update_freq: int) -> float:
    t0 = time.perf_counter()
    for i in range(20):
        step_fn(comp, i, actor_update_freq=actor_update_freq)
    return (time.perf_counter() - t0) * 1e3  # ms


# ── benchmark harness ──────────────────────────────────────────────────────────
def bench(comp, label: str, step_fn, actor_update_freq: int) -> float:
    print(f"\n── {label}")
    print(f"   warmup {N_WARMUP} blocks ...", end="", flush=True)
    for _ in range(N_WARMUP):
        run_block(comp, step_fn, actor_update_freq)
    print(" done")

    times = [run_block(comp, step_fn, actor_update_freq) for _ in range(N_BENCH)]

    med  = statistics.median(times)
    mean = statistics.mean(times)
    p10  = sorted(times)[int(N_BENCH * 0.1)]
    p90  = sorted(times)[int(N_BENCH * 0.9)]
    print(f"   {N_BENCH} blocks  |  "
          f"median {med:.1f} ms  mean {mean:.1f} ms  "
          f"p10 {p10:.1f} ms  p90 {p90:.1f} ms")
    print(f"   per gradient step (÷20):  {med/20:.2f} ms median")
    return med


# ── dropout variant ──────────────────────────────────────────────────────────
def add_dropout_to_critic(critic, p: float) -> None:
    """Insert nn.Dropout(p) after every activation in each critic MLP (in-place).

    MlpConfig has no dropout knob, so we patch the built nn.Sequential directly.
    Dropout carries no parameters, so weights/state_dict values are unchanged
    (only the Sequential indices shift).
    """
    for mlp in critic.mlp_list:
        if mlp.mlp is None:
            continue
        act = mlp.activation  # same instance is reused at each hidden layer
        new_layers = []
        for layer in mlp.mlp:
            new_layers.append(layer)
            if layer is act:
                new_layers.append(nn.Dropout(p))
        mlp.mlp = nn.Sequential(*new_layers)


def build_dropout_comp(comp, p: float):
    """Shallow-copy comp with critic + target + critic_opt replaced by dropout
    versions (same initial weights as the no-dropout critic for a fair compare)."""
    cfg = comp['cfg']

    def make_critic(src_state_dict):
        c = SacCritic(
            extractor_cls=comp['extractor_cls'],
            observation_space=comp['obs_space'],
            action_space=comp['action_space'],
            mlp_cfg=cfg.critic_mlp,
            num_critics=cfg.num_critics,
        ).to(DEVICE)
        # load weights BEFORE adding dropout — inserting Dropout shifts the
        # Sequential indices, so keys no longer match the plain critic.
        c.load_state_dict(src_state_dict)
        add_dropout_to_critic(c, p)
        return c

    critic_do = make_critic(comp['critic'].state_dict())
    target_do = make_critic(comp['critic'].state_dict())

    comp_do = dict(comp)
    comp_do['critic']        = critic_do
    comp_do['target_critic'] = target_do
    comp_do['critic_opt']    = torch.optim.Adam(critic_do.parameters(), lr=cfg.lr_q)
    return comp_do


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SAC-ZOP Training Block Benchmark")
    print(f"torch {torch.__version__}   device={DEVICE}")
    print("=" * 60)

    comp = setup()

    t_old = bench(comp, "OLD  freq=5  + else-branches", single_step_old, actor_update_freq=5)
    t_new = bench(comp, "NEW  freq=20 + no else-branches + cached alpha", single_step_new, actor_update_freq=20)

    saved_ms  = t_old - t_new
    saved_pct = saved_ms / t_old * 100
    print(f"\n{'='*60}")
    print(f"  OLD: {t_old:.1f} ms  →  NEW: {t_new:.1f} ms  (median per block)")
    print(f"  Speedup: {t_old/t_new:.2f}x   Saved: {saved_ms:+.1f} ms  ({saved_pct:+.1f}%)")

    # ── dropout overhead: same NEW block, critics with vs without dropout ────────
    comp_do = build_dropout_comp(comp, DROPOUT_P)
    t_do = bench(comp_do, f"NEW + dropout p={DROPOUT_P} (online + target critic)",
                 single_step_new, actor_update_freq=20)
    extra_ms  = t_do - t_new
    extra_pct = extra_ms / t_new * 100
    print(f"\n{'='*60}")
    print(f"  NEW no-dropout: {t_new:.1f} ms  →  NEW dropout: {t_do:.1f} ms  (median per block)")
    print(f"  Dropout overhead: {extra_ms:+.1f} ms  ({extra_pct:+.1f}%)   "
          f"per gradient step: {extra_ms/20:+.2f} ms")

    bench_critic_forward(comp)


def bench_critic_forward(comp, N: int = 2000):
    """Isolierter Vergleich: Python-Schleife vs vmap für den Critic-Forward."""
    from torch.func import vmap, functional_call
    from leap_c.torch.nn.scale import min_max_scaling

    critic  = comp['critic']
    cfg     = comp['cfg']
    o, a, *_ = comp['buf'].sample(cfg.batch_size)

    # ── Loop-Baseline ──────────────────────────────────────────────────────────
    for _ in range(50):
        critic(o, a)
    t0 = time.perf_counter()
    for _ in range(N):
        critic(o, a)
    t_loop = (time.perf_counter() - t0) / N * 1e6  # µs per forward

    # ── vmap-Version ──────────────────────────────────────────────────────────
    # Stack parameters of all critic MLPs along a new leading dim
    ref_mlp    = critic.mlp_list[0]
    all_params = [dict(mlp.named_parameters()) for mlp in critic.mlp_list]
    stacked    = {k: torch.stack([p[k] for p in all_params]).detach()
                  for k in all_params[0]}

    # Extractor is identity → features = observations
    a_norm = min_max_scaling(a, critic.action_space)
    xa = torch.cat([o, a_norm], dim=-1)  # [batch, obs+action]

    def single_fwd(params, x):
        return functional_call(ref_mlp, params, (x,))

    batched_fwd = vmap(single_fwd, in_dims=(0, None))

    for _ in range(50):
        batched_fwd(stacked, xa)
    t0 = time.perf_counter()
    for _ in range(N):
        batched_fwd(stacked, xa)
    t_vmap = (time.perf_counter() - t0) / N * 1e6  # µs per forward

    n = cfg.num_critics
    print(f"\n── Critic Forward isoliert  (n_critics={n}, batch={cfg.batch_size}, "
          f"hidden={list(cfg.critic_mlp.hidden_dims)})")
    print(f"   Loop : {t_loop:.1f} µs")
    print(f"   vmap : {t_vmap:.1f} µs")
    print(f"   → {t_loop/t_vmap:.2f}x  ({'lohnt sich' if t_loop/t_vmap > 1.3 else 'kein großer Gewinn'})")


if __name__ == "__main__":
    main()
