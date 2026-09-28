"""Deterministic benchmark of every trained simulation policy against the plain MPC.

Walks the checkpoint registry of the simulation runs (``runs*/<run>/checkpoints/actor.pth``),
rebuilds each policy, rolls it out ONCE per run in the hardware-matched sim env with
``deterministic=True`` (no exploration noise, no parameter sampling), and plots the
result in the same style as the real-hardware comparison (``real_acados/plot_trajectories.py``):

    Figure 1  accumulated reward over time, one line per run, coloured per controller
    Figure 2  x(t) and theta(t), one column per controller

The controller of a run is taken from its wandb config (``planner``), falling back to the
run-name convention ``<group>_<planner>_<reward>_s<seed>``. Runs from ``runs_sac`` have no
planner and are benchmarked as the pure-SAC baseline. The unlearned MPC (default
references, no MLP) is added as the reference baseline.

All rollouts use ONE reward function (``--reward``) regardless of what a run was trained
on, so the accumulated-reward curves are comparable across controllers.

Examples:
    # everything that exists, plain MPC baseline included
    python benchmark_policies.py

    # only the fresh 10k sweeps, write PDFs next to the runs
    python benchmark_policies.py --groups 8 9 10 11 --out benchmarks/sweep_8_11
"""

import sys
import re
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

# make 'bachelor' importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
import yaml

from bachelor.acados_cartpole.real_sac_zop.my_sac import SacActor, SacTrainerConfig
from bachelor.acados_cartpole.real_sac_zop.my_sac_zop import MpcSacActor, SacZopTrainerConfig
from bachelor.acados_cartpole.simulation_zaczop.planner_registry import make_planner
from bachelor.acados_cartpole.simulation_zaczop.sim_env import (
    RealCartPoleSimEnv,
    RealCartPoleSimConfig,
)
from bachelor.acados_cartpole.simulation_zaczop.rewards import get_reward_fn, REWARDS

from leap_c.planner import ControllerFromPlanner
from leap_c.torch.nn.extractor import get_extractor_cls

# stabilization criterion, identical to sim_sac_zop.py (200 steps at 10 ms = 2 s)
STABILIZATION_BUFFER_SIZE = 200
STABILIZATION_THRESHOLD = 0.15  # rad

# one colour per controller, kept muted like the real-hardware plots
COLORS = {
    "mpc": "#FF8C42",       # soft orange — the unlearned baseline
    "full": "#5EBA7D",      # soft green
    "du0": "#6A8CDB",       # soft blue
    "cart": "#C77DBB",      # soft purple
    "fullcart": "#E4B363",  # soft gold
    "sac": "#8C8C8C",       # grey — pure SAC, no MPC
}
LABELS = {
    "mpc": "MPC (unlearned)",
    "full": "SAC-ZOP full",
    "du0": "SAC-ZOP du0",
    "cart": "SAC-ZOP cart-only",
    "fullcart": "SAC-ZOP fullcart",
    "sac": "SAC (pure)",
}
# left-to-right order of the trajectory columns
ORDER = ["mpc", "full", "du0", "cart", "fullcart", "sac"]


def parse_args():
    p = ArgumentParser(description="Deterministic benchmark of trained sim policies vs. MPC")
    p.add_argument("--runs-dirs", nargs="+", type=Path,
                   default=[Path(__file__).parent / "runs", Path(__file__).parent / "runs_sac"],
                   help="Directories holding <run>/checkpoints/actor.pth")
    p.add_argument("--groups", nargs="+", default=None,
                   help="Only runs whose name starts with '<group>_' (e.g. 8 9 10 11)")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Only these seeds (parsed from the '_s<N>' suffix)")
    p.add_argument("--controllers", nargs="+", default=None, choices=ORDER,
                   help="Restrict to these controllers")
    p.add_argument("--reward", type=str, default="cos_bonus_only_theta", choices=sorted(REWARDS),
                   help="Reward used for ALL rollouts, so the curves stay comparable")
    p.add_argument("--dt", type=float, default=0.01, help="Sim time step [s] (runs used 10 ms)")
    p.add_argument("--max-ep-steps", type=int, default=1000,
                   help="Rollout length in env steps (1000 at 10 ms = 10 s)")
    p.add_argument("--eval-seed", type=int, default=0,
                   help="Env reset seed — the sim reset is deterministic, kept for completeness")
    p.add_argument("--no-mpc", action="store_true", help="Skip the unlearned-MPC baseline")
    p.add_argument("--mpc-planner", type=str, default="full",
                   help="Which OCP the unlearned-MPC baseline solves")
    p.add_argument("--t-max", type=float, default=None,
                   help="Clip the plots at this time [s] (default: full rollout)")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "benchmarks",
                   help="Where the PDFs and the summary CSV are written")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


# discovery ------------------------------------------------------------------
def run_config(run_dir: Path) -> dict:
    """Best-effort read of a run's wandb config.yaml (planner, d_max, reward, ...)."""
    cfgs = sorted(run_dir.glob("wandb/*/files/config.yaml"))
    if not cfgs:
        return {}
    try:
        raw = yaml.safe_load(cfgs[0].read_text()) or {}
    except Exception:
        return {}
    # wandb stores every entry as {key: {value: ...}}
    return {k: v["value"] for k, v in raw.items()
            if isinstance(v, dict) and "value" in v}


def discover(args) -> list[dict]:
    """Collect every run that has a usable actor checkpoint."""
    runs = []
    for base in args.runs_dirs:
        if not base.is_dir():
            print(f"  (skipping missing {base})")
            continue
        pure_sac = "sac" in base.name  # runs_sac / runs_sac_trainer hold pure-SAC runs
        for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            ckpt = run_dir / "checkpoints" / "actor.pth"
            if not ckpt.is_file():
                continue
            name = run_dir.name
            if args.groups and not any(name.startswith(f"{g}_") for g in args.groups):
                continue
            m = re.search(r"_s(\d+)$", name)
            seed = int(m.group(1)) if m else None
            if args.seeds is not None and seed not in args.seeds:
                continue

            cfg = run_config(run_dir)
            if pure_sac:
                controller = "sac"
            else:
                controller = cfg.get("planner")
                if not controller:  # fall back to the <group>_<planner>_... convention
                    parts = name.split("_")
                    controller = next((p for p in parts if p in ORDER and p != "sac"), None)
                if not controller:
                    print(f"  (skipping {name}: planner not identifiable)")
                    continue
            if args.controllers and controller not in args.controllers:
                continue

            runs.append({
                "name": name, "dir": run_dir, "ckpt": ckpt, "seed": seed,
                "controller": controller, "d_max": cfg.get("d_max"),
                "trained_reward": cfg.get("reward"),
            })
    return runs


# policy construction --------------------------------------------------------
_planner_cache: dict[tuple, tuple] = {}


def get_controller(planner_name: str, d_max, export_root: Path):
    """Build (and cache) a planner + its ControllerFromPlanner wrapper."""
    key = (planner_name, None if d_max is None else float(d_max))
    if key not in _planner_cache:
        kwargs = {}
        if planner_name == "du0" and d_max is not None:
            kwargs["d_max"] = float(d_max)
        tag = planner_name + ("" if d_max is None else f"_d{d_max:g}")
        planner, _ = make_planner(planner_name, export_root / f"acados_{tag}", **kwargs)
        _planner_cache[key] = (planner, ControllerFromPlanner(planner))
    return _planner_cache[key]


def build_policy(run: dict, obs_space, Fmax: float, export_root: Path, device: str):
    """Returns a callable ``step(obs_batch, ctx) -> (force, ctx)`` for this run."""
    ctrl_name = run["controller"]

    if ctrl_name == "sac":
        action_space = gym.spaces.Box(low=-Fmax, high=Fmax, shape=(1,), dtype=np.float32)
        cfg = SacTrainerConfig()
        cfg.critic_mlp.norm_layer = "layer_norm"
        actor = SacActor(
            extractor_cls=get_extractor_cls("identity"),
            action_space=action_space,
            observation_space=obs_space,
            distribution_name=cfg.distribution_name,
            mlp_cfg=cfg.actor_mlp,
        ).to(device)
        actor.load_state_dict(torch.load(run["ckpt"], map_location=device))
        actor.eval()

        def step(obs_batch, ctx):
            with torch.no_grad():
                action, _, _ = actor(obs_batch, deterministic=True)
            return float(action[0].cpu().numpy().squeeze()), None

        return step, None

    planner, controller_wrapped = get_controller(ctrl_name, run["d_max"], export_root)
    cfg = SacZopTrainerConfig()
    cfg.critic_mlp.norm_layer = "layer_norm"
    actor = MpcSacActor(
        extractor_cls=get_extractor_cls("identity"),
        observation_space=obs_space,
        controller=controller_wrapped,
        distribution_name=cfg.distribution_name,
        mlp_cfg=cfg.actor_mlp,
        init_param_with_default=cfg.init_param_with_default,
    ).to(device)
    actor.load_state_dict(torch.load(run["ckpt"], map_location=device))
    actor.eval()

    def step(obs_batch, ctx):
        with torch.no_grad():
            out = actor(obs_batch, ctx, deterministic=True)
        return float(out.action[0].cpu().numpy().squeeze()), out.ctx

    def init_ctx(obs_batch):
        with torch.no_grad():
            ctx_planner, _, _, _, _ = planner(obs_batch, ctx=None)
            out = actor(obs_batch, ctx_planner, deterministic=True)
        return out.ctx

    return step, init_ctx


def build_mpc_baseline(planner_name: str, export_root: Path, device: str):
    """The unlearned MPC: fixed default references, no MLP, no noise."""
    planner, controller_wrapped = get_controller(planner_name, None, export_root)
    dp = np.asarray(controller_wrapped.default_param(None), dtype=np.float32)
    default_param = torch.as_tensor(dp, device=device).reshape(1, -1)

    def step(obs_batch, ctx):
        with torch.no_grad():
            ctx, action = controller_wrapped(obs_batch, default_param, ctx=ctx)
        return float(action[0].cpu().numpy().squeeze()), ctx

    def init_ctx(obs_batch):
        with torch.no_grad():
            ctx_planner, _, _, _, _ = planner(obs_batch, ctx=None)
            ctx, _ = controller_wrapped(obs_batch, default_param, ctx=ctx_planner)
        return ctx

    return step, init_ctx


# rollout --------------------------------------------------------------------
def rollout(step_fn, init_ctx, env, args, device: str) -> dict:
    """One deterministic episode. Returns the recorded trajectory."""
    obs, _ = env.reset(seed=args.eval_seed)
    ctx = None
    if init_ctx is not None:
        ctx = init_ctx(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))

    t, xs, thetas, forces, rewards = [], [], [], [], []
    cum, tripped, stabilized_at = 0.0, False, None
    theta_hist: list[float] = []

    for k in range(args.max_ep_steps):
        obs_batch = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        force, ctx = step_fn(obs_batch, ctx)
        obs, reward, terminated, truncated, info = env.step(force)
        cum += reward

        t.append((k + 1) * args.dt)
        xs.append(float(obs[0]))
        thetas.append(float(obs[1]))
        forces.append(force)
        rewards.append(cum)

        theta_hist.append(float(obs[1]))
        if stabilized_at is None and len(theta_hist) >= STABILIZATION_BUFFER_SIZE:
            window = theta_hist[-STABILIZATION_BUFFER_SIZE:]
            if all(abs(a) <= STABILIZATION_THRESHOLD for a in window):
                stabilized_at = (k + 1) * args.dt

        if terminated:
            tripped = True
            break
        if truncated:
            break

    return {
        "t": np.array(t), "x": np.array(xs),
        # unwrapped like the real-hardware CSVs, so swingups read as a smooth curve
        "theta": np.unwrap(np.array(thetas)),
        "force": np.array(forces), "reward": np.array(rewards),
        "cum_reward": cum, "tripped": tripped,
        "stabilized_at": stabilized_at, "steps": len(t),
    }


# plotting -------------------------------------------------------------------
def make_plots(results: list[dict], args, out_dir: Path):
    present = [c for c in ORDER if any(r["controller"] == c for r in results)]
    t_max = args.t_max or max(r["traj"]["t"][-1] for r in results)

    def clip(traj, key):
        m = traj["t"] <= t_max
        return traj["t"][m], traj[key][m]

    # --- Figure 1: accumulated reward -------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    seen = set()
    for r in results:
        c = r["controller"]
        tt, rr = clip(r["traj"], "reward")
        ax1.plot(tt, rr, color=COLORS[c], alpha=0.9, linewidth=1.5,
                 label=LABELS[c] if c not in seen else "")
        seen.add(c)
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel(f"Accumulated reward ({args.reward})", fontsize=12)
    ax1.grid(True, alpha=0.3, which="major")
    ax1.grid(True, alpha=0.15, which="minor", linestyle=":")
    ax1.minorticks_on()
    ax1.legend(fontsize=10, loc="best")
    ax1.set_xlim(0, t_max)
    ax1.set_title(f"Deterministic rollout — accumulated reward (dt = {args.dt*1000:.0f} ms)",
                  fontsize=13)
    fig1.tight_layout()
    p1 = out_dir / "benchmark_reward_comparison.pdf"
    fig1.savefig(p1, format="pdf", dpi=150, bbox_inches="tight")
    print(f"  wrote {p1}")

    # --- Figure 2: x(t) and theta(t), one column per controller ------------
    n = len(present)
    fig2, axes = plt.subplots(2, n, figsize=(4.2 * n, 8), sharex=True, squeeze=False)
    all_x = np.concatenate([clip(r["traj"], "x")[1] for r in results])
    all_th = np.concatenate([clip(r["traj"], "theta")[1] for r in results])
    x_lim = (all_x.min() - 0.1 * np.ptp(all_x), all_x.max() + 0.1 * np.ptp(all_x))
    th_lim = (all_th.min() - 0.1 * np.ptp(all_th), all_th.max() + 0.1 * np.ptp(all_th))

    for col, c in enumerate(present):
        runs_c = [r for r in results if r["controller"] == c]
        ax_x, ax_th = axes[0][col], axes[1][col]
        for r in runs_c:
            tt, xx = clip(r["traj"], "x")
            ax_x.plot(tt, xx, color=COLORS[c], alpha=0.85, linewidth=1.3)
            tt, th = clip(r["traj"], "theta")
            ax_th.plot(tt, th, color=COLORS[c], alpha=0.85, linewidth=1.3)
        ax_x.set_title(f"{LABELS[c]}  (n={len(runs_c)})", fontsize=12)
        ax_x.set_ylim(x_lim)
        ax_th.set_ylim(th_lim)
        ax_th.set_xlim(0, t_max)
        ax_th.set_xlabel("Time (s)", fontsize=11)
        for ax in (ax_x, ax_th):
            ax.grid(True, alpha=0.3, which="major")
            ax.grid(True, alpha=0.15, which="minor", linestyle=":")
            ax.minorticks_on()
        if col == 0:
            ax_x.set_ylabel("x (m)", fontsize=12)
            ax_th.set_ylabel("θ (rad, unwrapped)", fontsize=12)

    fig2.suptitle(f"Deterministic rollouts per controller (dt = {args.dt*1000:.0f} ms, "
                  f"reward = {args.reward})", fontsize=13)
    fig2.tight_layout()
    p2 = out_dir / "benchmark_trajectories.pdf"
    fig2.savefig(p2, format="pdf", dpi=150, bbox_inches="tight")
    print(f"  wrote {p2}")


def write_summary(results: list[dict], out_dir: Path):
    csv_path = out_dir / "benchmark_summary.csv"
    with csv_path.open("w") as fh:
        fh.write("controller,run,seed,cum_reward,steps,tripped,stabilized_at_s,max_abs_force\n")
        for r in results:
            tr = r["traj"]
            st = "" if tr["stabilized_at"] is None else f"{tr['stabilized_at']:.3f}"
            fh.write(f"{r['controller']},{r['name']},{r['seed']},{tr['cum_reward']:.4f},"
                     f"{tr['steps']},{int(tr['tripped'])},{st},"
                     f"{np.abs(tr['force']).max():.3f}\n")
    print(f"  wrote {csv_path}")

    by = defaultdict(list)
    for r in results:
        by[r["controller"]].append(r["traj"])
    print("\n" + "=" * 78)
    print(f"{'controller':<18}{'n':>3}{'cum reward (mean±std)':>26}{'tripped':>10}{'stabilized':>12}")
    print("-" * 78)
    for c in ORDER:
        if c not in by:
            continue
        trs = by[c]
        cums = np.array([t["cum_reward"] for t in trs])
        trip = sum(t["tripped"] for t in trs)
        stab = sum(t["stabilized_at"] is not None for t in trs)
        print(f"{LABELS[c]:<18}{len(trs):>3}{cums.mean():>15.2f} ± {cums.std():<8.2f}"
              f"{trip:>8}/{len(trs):<2}{stab:>9}/{len(trs):<2}")
    print("=" * 78)


def main():
    args = parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    export_root = out_dir / "_acados"
    export_root.mkdir(parents=True, exist_ok=True)

    print("Discovering runs...")
    runs = discover(args)
    if not runs and args.no_mpc:
        print("No runs found and MPC baseline disabled — nothing to do.")
        return
    print(f"  found {len(runs)} runs with checkpoints")

    sim_cfg = RealCartPoleSimConfig(dt=args.dt)
    reward_fn = get_reward_fn(args.reward)
    env = RealCartPoleSimEnv(reward_fn, cfg=sim_cfg, max_episode_steps=args.max_ep_steps)
    obs_space = env.observation_space
    Fmax = float(sim_cfg.Fmax)

    results = []

    if not args.no_mpc and (not args.controllers or "mpc" in args.controllers):
        print(f"Rolling out unlearned MPC baseline (planner={args.mpc_planner})...")
        try:
            step_fn, init_ctx = build_mpc_baseline(args.mpc_planner, export_root, args.device)
            traj = rollout(step_fn, init_ctx, env, args, args.device)
            results.append({"controller": "mpc", "name": f"mpc_{args.mpc_planner}",
                            "seed": None, "traj": traj})
            print(f"  cum reward = {traj['cum_reward']:.2f}, steps = {traj['steps']}, "
                  f"tripped = {traj['tripped']}")
        except Exception as e:
            print(f"  MPC baseline failed: {e}")

    for run in runs:
        print(f"Rolling out {run['name']} ({run['controller']})...")
        try:
            step_fn, init_ctx = build_policy(run, obs_space, Fmax, export_root, args.device)
            traj = rollout(step_fn, init_ctx, env, args, args.device)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        results.append({"controller": run["controller"], "name": run["name"],
                        "seed": run["seed"], "traj": traj})
        print(f"  cum reward = {traj['cum_reward']:.2f}, steps = {traj['steps']}, "
              f"tripped = {traj['tripped']}")

    if not results:
        print("Nothing rolled out successfully.")
        return

    print("\nPlotting...")
    make_plots(results, args, out_dir)
    write_summary(results, out_dir)


if __name__ == "__main__":
    main()
