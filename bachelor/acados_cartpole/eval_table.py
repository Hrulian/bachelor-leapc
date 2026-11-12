"""LaTeX eval table: reward, safety and solver figures of the benchmark controllers.

For each controller of `plot_benchmark` and plain SAC N_RUNS eval runs are scored,
each cell being mean +- std over the runs. The reward is symmetric in theta and
does not depend on what a controller was trained on, so all of them are scored on
the same scale:

    linear (default):  r = 1 - |theta| / pi        the swing-up term of
                       `reward_default` in simulation_zaczop/rewards.py
    cos:               r = 0.5 * (1 + cos theta)   the upright term of cos_bonus_spin3

Both are in [0, 1], 1 = upright, 0 = hanging. It is taken every 50 ms, from the
sample closest to that instant (every fifth sample at 10 ms), as if every run had
been logged every 50 ms, and summed over the whole EPISODE_S episode, the first
SPLIT seconds and the rest. A run upright for all 10 s scores 200.

The other columns:

    Solver Success  share of the control steps whose acados solve converged, after
                    the retry (`solve_success`).
    Inference Time  time per control step: the acados solve with its retry
                    (`solve_time_ms`) and, in the steps calling the actor, the SAC
                    network. The logs do not have the network's time, so it is
                    measured here from the checkpoint, on this machine. The Python
                    overhead around the solve is left out: only the MPC logs it
                    (`host_solve_ms`), so it cannot be compared.

Plain SAC has no solver: no solver success, and its inference time is
the network alone, one measured time without a std.

One more column comes from the training instead of the eval runs, so it exists
for the learning controllers only: the wandb run that produced the evaluated
checkpoint (`wandb_run_id` in its meta.pth) gives the constraint violations in
the first TRAINING_STEPS RL steps (`terminations/total_e`). One training run each, so no
std. The violations are not read at the end of the run: runs go on for different
lengths past that, and a resumed one restarts the counter (SAC_4: from 74 to 0).

The runs are the ones of `plot_benchmark.pick_runs`: in run order, those that finished
their episode first. A run that tripped or was truncated is used only if the controller
has no further finished run to take its place; it then scores what it collected until it
ended, nothing after. A run with a large gap in its log is not used at all.

    python -m bachelor.acados_cartpole.eval_table
    python -m bachelor.acados_cartpole.eval_table --reward cos
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from gymnasium import spaces
from torch import nn

from bachelor.acados_cartpole.plot_benchmark import (
    EPISODE_S,
    HERE,
    MAX_GAP_SAMPLES,
    N_RUNS,
    TABLE_CONTROLLERS,
    pick_runs,
)
from bachelor.acados_cartpole.plot_eval_runs import DEFAULT_CUM_REF_DT
from bachelor.acados_cartpole.plot_training_curves import ENTITY, GROUPS, NAMES, fetch_seed
from leap_c.torch.nn.bounded_distributions import SquashedGaussian
from leap_c.torch.nn.mlp import Mlp, MlpConfig

# Controller label -> wandb project of its training runs, cached like in `plot_training_curves`.
TRAINING_PROJECTS = {NAMES[label]: project for label, project, *_ in GROUPS}
TRAINING_CACHE = HERE / "paper_plots" / "wandb_cache"
TRAINING_STEPS = 10_000

# Rounds of forward passes `actor_forward_ms` takes the median of per round.
N_TIMED_ROUNDS = 10
N_TIMED_CALLS = 200

# name -> (reward of the wrapped angle, formula for the caption)
REWARDS = {
    "linear": (lambda theta: 1.0 - np.abs(theta) / np.pi, r"$r = 1 - |\theta|/\pi$"),
    "cos": (lambda theta: 0.5 * (1.0 + np.cos(theta)), r"$r = \tfrac12(1+\cos\theta)$"),
}

# Controller label -> its macro in main_paper.tex
MACROS = {
    "MPC (10 ms)": r"\mpcten",
    "MPC (50 ms)": r"\mpcfifty",
    "Angle-10": r"\zopangleten",
    "Angle": r"\zopangle",
    "Pos": r"\zoppos",
    "Cart": r"\zopcart",
}

# Columns of `metrics`: (two-line header, decimals, +1 if higher is better, -1 if lower is)
COLUMNS = (
    (("Total", "Reward"), 1, +1),
    (("Swing-up", "Reward"), 1, +1),
    (("Balancing", "Reward"), 1, +1),
    (("Solver", r"Success (\%)"), 1, +1),
    (("Inference", "Time (ms)"), 2, -1),
)


def wrap(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def sampled_rewards(t: np.ndarray, r: np.ndarray, n_samples: int) -> np.ndarray:
    """The reward at the instants k * DEFAULT_CUM_REF_DT, from the sample closest to
    each; 0 at the instants after the run's end."""
    grid = np.arange(n_samples) * DEFAULT_CUM_REF_DT
    closest = np.abs(t[:, None] - grid).argmin(axis=0)
    return np.where(grid <= t[-1] + DEFAULT_CUM_REF_DT / 2, r[closest], 0.0)


def actor_forward_ms(checkpoint: Path) -> float:
    """Time of one pass through the SAC network on the CPU, the part of
    `MpcSacActor.forward` before it calls the controller. The smallest median of
    several rounds: the machine at times doubles a whole round's time for a moment,
    which says nothing about the network."""
    state = torch.load(checkpoint, map_location="cpu")
    n_obs = state["mlp.mlp.0.weight"].shape[1]
    n_param = state["bounded_distribution.loc"].shape[0]
    mlp = Mlp(n_obs, [n_param, n_param], MlpConfig())
    distribution = SquashedGaussian(spaces.Box(-1.0, 1.0, (n_param,)))
    nn.ModuleDict({"mlp": mlp, "bounded_distribution": distribution}).load_state_dict(state)

    obs = torch.zeros(1, n_obs)
    times = []
    with torch.no_grad():
        for _ in range(N_TIMED_ROUNDS * N_TIMED_CALLS):
            start = time.perf_counter()
            distribution(*mlp(obs), deterministic=True)
            times.append(time.perf_counter() - start)
    rounds = np.reshape(times, (N_TIMED_ROUNDS, N_TIMED_CALLS))
    return float(np.median(rounds, axis=1).min()) * 1000.0


def training_violations(meta_path: Path, project: str) -> int:
    """The constraint violations in the first TRAINING_STEPS of the wandb run that
    produced a checkpoint."""
    run_id = torch.load(meta_path, map_location="cpu", weights_only=False)["wandb_run_id"]
    api = wandb.Api()
    run = api.run(f"{ENTITY or api.default_entity}/{project}/{run_id}")
    seed = fetch_seed(run, TRAINING_CACHE / project, refresh=False)
    episodes = seed.episodes[seed.episodes["_step"] <= TRAINING_STEPS]
    return int(episodes["terminations/total_e"].iloc[-1])


def metrics(picked, reward, split: float, actor_ms: float) -> np.ndarray:
    """One row per picked run with the columns of COLUMNS. `actor_ms` is the time of
    one actor call, 0 for the MPC. A log without `actor_called` calls it every step,
    one without solver columns (plain SAC) has a NaN success."""
    n_samples = int(round(EPISODE_S / DEFAULT_CUM_REF_DT))
    n_split = int(round(split / DEFAULT_CUM_REF_DT))
    out = []
    for _, df, _ in picked:
        df = df[df["time_s"] <= EPISODE_S]
        r = sampled_rewards(df["time_s"].to_numpy(dtype=float),
                            reward(wrap(df["theta_rad"].to_numpy(dtype=float))), n_samples)
        inference = df.get("solve_time_ms", 0.0) + actor_ms * df.get("actor_called", 1.0)
        out.append((r.sum(), r[:n_split].sum(), r[n_split:].sum(),
                    100.0 * np.mean(df.get("solve_success", np.nan)), np.mean(inference)))
    return np.array(out)


def latex_table(rows, caption: str, label: str) -> str:
    """`rows` are (controller, mean, std, violations), mean and std per column, violations
    the result of `training_violations` or None without a training. A NaN mean is a cell
    without data, a NaN std a value without a spread."""
    means = np.array([mean for _, mean, _, _ in rows])
    best = [np.nanmax(side * means[:, i]) * side for i, (_, _, side) in enumerate(COLUMNS)]
    headers = [header for header, _, _ in COLUMNS]
    headers = [("", "Controller"), *headers, ("Constr.", "Viol.")]
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{lccccccc}",
        r"    \toprule",
        r"     & \multicolumn{5}{c}{Evaluation} & \multicolumn{1}{c}{Training} \\",
        r"    \cmidrule(lr){2-6} \cmidrule(lr){7-7}",
        "    " + " & ".join(top for top, _ in headers) + r" \\",
        "    " + " & ".join(bottom for _, bottom in headers) + r" \\",
        r"    \midrule",
    ]
    for name, mean, std, violations in rows:
        cells = []
        for (_, decimals, _), m, s, b in zip(COLUMNS, mean, std, best):
            if np.isnan(m):
                cells.append("--")
                continue
            value = f"{m:.{decimals}f}"
            # compared as printed, so that two cells showing the same number are both bold
            value = rf"\mathbf{{{value}}}" if value == f"{b:.{decimals}f}" else value
            cells.append(f"${value}$" if np.isnan(s) else rf"${value} \pm {s:.{decimals}f}$")
        cells.append("--" if violations is None else str(violations))
        lines.append(f"    {MACROS.get(name, name):<14} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=HERE / "paper_plots" / "eval_table.tex",
                        help="Output file (default: paper_plots/eval_table.tex).")
    parser.add_argument("--reward", choices=sorted(REWARDS), default="linear",
                        help="Symmetric reward to score with (default: linear).")
    parser.add_argument("--runs", type=int, default=N_RUNS,
                        help=f"Runs per controller (default: {N_RUNS}).")
    parser.add_argument("--split", type=float, default=5.0,
                        help="End of the swing-up window in seconds (default: 5).")
    parser.add_argument("--min-duration", type=float, default=9.5,
                        help="Runs shorter than this many seconds count as failed (default: 9.5).")
    args = parser.parse_args()
    reward, formula = REWARDS[args.reward]

    rows, failed, replaced, gaps, short, network_ms = [], [], [], [], [], None
    for c in TABLE_CONTROLLERS:
        picked, swapped, gapped = pick_runs(c, args.runs, args.min_duration)
        if len(picked) < args.runs:
            print(f"warning: {c.label} has only {len(picked)} runs")
            short.append(f"{MACROS.get(c.label, c.label)}: {len(picked)}")
        used = [(path, reason) for path, _, reason in picked if reason is not None]
        for path, reason in swapped:
            print(f"{c.label}: replaced {path.relative_to(c.directory)} ({reason})")
        for path, reason in used:
            print(f"{c.label}: kept {path.relative_to(c.directory)} ({reason}), no spare run")
        for path in gapped:
            print(f"{c.label}: dropped {path.relative_to(c.directory)} (log gap)")
        print(f"{c.label}: " + ", ".join(path.stem for path, _, _ in picked))
        if used:
            failed.append(f"{MACROS.get(c.label, c.label)}: {len(used)}")
        if swapped:
            replaced.append(f"{MACROS.get(c.label, c.label)}: {len(swapped)}")
        if gapped:
            gaps.append(f"{MACROS.get(c.label, c.label)}: {len(gapped)}")

        actor_ms, violations = 0.0, None
        checkpoint = picked[0][0].parent
        if (checkpoint / "actor.pth").exists():
            # timed once: the networks of all learning controllers have the same size
            network_ms = network_ms or actor_forward_ms(checkpoint / "actor.pth")
            actor_ms = network_ms
            violations = training_violations(checkpoint / "meta.pth", TRAINING_PROJECTS[c.label])
            print(f"{c.label}: training {violations} violations")

        s = metrics(picked, reward, args.split, actor_ms)
        std = s.std(axis=0, ddof=1)
        if "solve_time_ms" not in picked[0][1]:
            std[-1] = np.nan  # the one measured network time, no spread over the runs
        rows.append((c.label, s.mean(axis=0), std, violations))

    notes = ""
    if replaced:
        notes += (" Tripped runs were replaced by further runs of the same controller "
                  f"({', '.join(replaced)}).")
    if failed:
        notes += (" Where no further run was available, the tripped run counts with the "
                  f"reward collected until the trip ({', '.join(failed)}).")
    if gaps:
        notes += (f" Runs with a gap of more than {MAX_GAP_SAMPLES} samples in the log were "
                  f"replaced as well ({', '.join(gaps)}).")

    split = f"{args.split:g}"
    runs = f"{args.runs} eval runs per controller" + (f" ({', '.join(short)})" if short else "")
    caption = (
        rf"Evaluation on hardware, mean $\pm$ std over {runs}. "
        rf"Symmetric reward {formula} in $[0, 1]$, taken every 50~ms and summed over the "
        rf"10~s episode (maximum 200), the first {split}~s (swing-up) and the remaining "
        r"time (balancing). Solver success: "
        r"control steps with a converged acados solve. Inference time: mean per control step "
        r"of the acados solve and, for SAC-ZOP, the SAC network "
        rf"({network_ms:.2f}~ms per call, every step at 50~ms, every fifth at 10~ms), "
        r"for SAC the network alone. "
        r"Training: the run that produced the evaluated controller, with its constraint "
        rf"violations in the first {TRAINING_STEPS // 1000}k RL steps (the MPC is not trained)."
        + notes
    )
    table = latex_table(rows, caption, f"tab:eval_{args.reward}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + table)
    args.out.write_text(table)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
