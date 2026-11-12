"""Training-metric figures for the paper, straight from wandb.

Pulls the training history of every seed in the groups below, aligns the seeds
on a common RL-step axis and draws the mean across seeds per group. Every figure
is written twice, once per shaded region around the mean: `minmax` (min-max over
the seeds) and `std` (mean +- std).

    python -m bachelor.acados_cartpole.plot_training_curves
    python -m bachelor.acados_cartpole.plot_training_curves --bin 250 --x steps
    python -m bachelor.acados_cartpole.plot_training_curves --refresh --format png pdf
    python -m bachelor.acados_cartpole.plot_training_curves --paper --format pdf png

THE X AXIS IS BUILT FROM THE RL STEP (`_step`), NOT THE ENV STEP. One RL step is
one actor decision = 50 ms of real interaction in every variant (the 10 ms K-step
runs hold the action for N=5 MPC steps), so the K-step runs and the 50 ms runs
share the axis. The per-episode metrics are already logged in RL steps for the
same reason (see the `/ N` in real_sac_zop.py). By default the axis is drawn as
interaction time in minutes (RL step * 50 ms); `--x steps` shows the raw steps.

Per-episode metrics (`*_e`) sit at a different `_step` in every seed, because
the episodes have different lengths, so they cannot be averaged point by point.
Instead the step axis is cut into windows of `--bin` RL steps: each seed is
averaged inside a window (an episode belongs to the window it ENDS in), then
mean and band are taken across the seeds per window. Per-step metrics (`*_s`)
get the same treatment, averaging every RL step inside the window.

`stabilization/achieved_at_step_e` only exists for episodes that stabilized.
By default (`--achieved censored`) a failed episode counts as the full episode
length, i.e. "not within 10 s"; that keeps the failures in the average instead
of hiding them. `--achieved successes` averages the successful episodes only.

MPC needs no training, so its eval mean (the CSVs `plot_eval_runs` reads) is
drawn as a dashed horizontal reference wherever the metric exists for MPC, over
the runs of the eval table (`plot_benchmark.pick_runs`). Eval
returns are rescaled by the NOMINAL period / 50 ms (x0.2 at 10 ms), the same
`/ N` the training logs use. plot_paper scales by the measured median period,
which jitter pulls to ~8.9 ms at 10 ms, so its 10 ms reference comes out ~11 %
lower than this one.

"Constraint violations" are the logged terminations: the episode ended because
the cart left the track limit (|x| > X_TERM_M) or the hardware safety tripped.
Reaching the episode length does not count.

The wandb history is cached as CSV below `--cache`; `--refresh` downloads again.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import wandb

from bachelor.acados_cartpole.plot_eval_runs import DEFAULT_ROOTS, discover
from bachelor.acados_cartpole.plot_paper import load_eval_runs
from leap_c.utils.latexify import latex_plot_context

HERE = Path(__file__).resolve().parent

RL_DT = 0.05  # s per RL step, every variant
MAX_EP_RL_STEPS = 200  # 10 s episode, in RL steps, every variant

ENTITY = None  # None = your wandb default entity

# label, project, wandb group, run names (None = every run in the group).
# Labels and colours match learning_curves.png from plot_paper.
GROUPS = [
    # SAC_3 stopped at 8120 RL steps (motor overheated) - left out, the paper mentions it
    ("real_sac", "Paper-Real-SAC", "main", {"SAC_0", "SAC_1", "SAC_2", "SAC_4"}),
    ("real_sac_zop_full", "Paper-Real_SAC-ZOP", "full",
     {"full_ocp_s0", "full_ocp_s1", "full_ocp_s5", "full_ocp_s8", "full_ocp_s9"}),
    ("real_sac_zop_cart", "Paper-Real_SAC-ZOP", "cart",
     {"cart_ocp_s1", "cart_ocp_s2", "cart_ocp_s3", "cart_ocp_s4", "cart_ocp_s5"}),
    ("real_sac_zop_fullcart", "Paper-Real_SAC-ZOP", "fullcart", None),
    ("real_sac_zop_full_10ms", "Paper-Real_SAC-ZOP", "K-step", None),
]
# legend names in the figures
NAMES = {
    "real_sac": "SAC",
    "real_sac_zop_full": "Angle",
    "real_sac_zop_cart": "Cart",
    "real_sac_zop_fullcart": "Pos",
    "real_sac_zop_full_10ms": "Angle-10",
}
COLORS = {label: plt.cm.tab10(i) for label, i in [
    ("real_sac", 2), ("real_sac_zop_cart", 3), ("real_sac_zop_full", 4),
    ("real_sac_zop_full_10ms", 5), ("real_sac_zop_fullcart", 6),
]}

# MPC eval references: legend name, eval directory, nominal control period (s), colour.
# Colours match the MPC lines in learning_curves.png.
MPC_REFS = [
    ("MPC-10", "mpc_full_10ms_saz", 0.01, plt.cm.tab10(0)),
    ("MPC-50", "mpc_full_50ms_saz", 0.05, plt.cm.tab10(1)),
]
# same selection as eval_table
MPC_N_RUNS = 20
MPC_MIN_DURATION = 9.5  # s

EPISODE_KEYS = [
    "episode/cumulative_reward_e",
    "episode/stabilized_e",
    "episode/longest_upright_steps_e",
    "episode/steps_e",
    "terminations/total_e",
]
ACHIEVED_KEY = "stabilization/achieved_at_step_e"  # sparse: successes only
STEP_KEYS = ["step/reward_s", "step/upright_s", "step/pi_success_rate_s"]

# The figure of this script in the paper, with its band.
PAPER_PAIR = "training_return_stabilization"
PAPER_BAND = "minmax"


# Text width of the paper (l4dc2027.cls -> jmlr.cls, one column, letter paper), in inches.
# The paper figures are drawn at this width and included at `width=\textwidth`, so
# every figure shows the fonts at the sizes below, unscaled.
TEXTWIDTH_IN = 6.0
FONT_PT = 9  # labels, titles, legend; \footnotesize of the 11 pt body
TICK_PT = 8


def paper_font():
    """The fonts and sizes of the paper figures, the same in every figure: the paper loads
    `times`, which sets the text in Times (`ptm`) and leaves the math in Computer Modern.
    Not the `Times` of `font.serif`, which would bring mathptmx. Lines, ticks and spines
    are thin to match, since the figures are drawn at their printed size."""
    preamble = r"\usepackage{gensymb} \usepackage{amsmath} \renewcommand{\rmdefault}{ptm}"
    return latex_plot_context(**{
        "text.latex.preamble": preamble,
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT, "axes.titlesize": FONT_PT, "legend.fontsize": FONT_PT,
        "figure.titlesize": FONT_PT,
        "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT,
        "axes.linewidth": 0.6, "grid.linewidth": 0.5, "lines.linewidth": 1.2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.titlepad": 3, "axes.labelpad": 2,
    })


@dataclass
class Metric:
    key: str
    title: str
    ylabel: str
    per_episode: bool
    scale: float = 1.0  # applied to the logged value, e.g. RL steps -> s
    cumulative: bool = False  # a running counter: hold the last value, never smooth
    ylim: tuple[float, float] | None = None


METRICS = [
    Metric("episode/cumulative_reward_e", "Training reward", "Training return", True),
    Metric("episode/stabilized_e", "Stabilization rate", r"Stabilized episodes [\%]",
           True, scale=100.0, ylim=(-3, 103)),
    Metric(ACHIEVED_KEY, "Time to stabilize", "time into episode [s]", True, scale=RL_DT,
           ylim=(0, MAX_EP_RL_STEPS * RL_DT * 1.03)),
    Metric("terminations/total_e", "Constraint violations", "cumulative violations", True,
           cumulative=True),
    Metric("step/reward_s", "Step reward", "reward per RL step", False),
    Metric("step/pi_success_rate_s", "MPC solver success", "solver success rate", False,
           ylim=None),
]


@dataclass
class Seed:
    name: str
    episodes: pd.DataFrame  # one row per episode, indexed by _step at episode end
    steps: pd.DataFrame  # one row per RL step


@dataclass
class Group:
    label: str
    seeds: list[Seed] = field(default_factory=list)


# READING ######################################################################
def _scan(run, keys: list[str]) -> pd.DataFrame:
    """History rows that hold ALL of `keys` (wandb drops rows missing any)."""
    rows = run.scan_history(keys=["_step", *keys], page_size=10_000)
    df = pd.DataFrame(list(rows))
    if df.empty:
        return pd.DataFrame(columns=["_step", *keys])
    return df.sort_values("_step").drop_duplicates("_step")


def fetch_seed(run, cache: Path, refresh: bool) -> Seed:
    ep_file = cache / f"{run.id}_episodes.csv"
    st_file = cache / f"{run.id}_steps.csv"
    if refresh or not (ep_file.exists() and st_file.exists()):
        print(f"    downloading {run.name} ({run.id})")
        episodes = _scan(run, EPISODE_KEYS)
        achieved = _scan(run, [ACHIEVED_KEY])
        episodes = episodes.merge(achieved, on="_step", how="left")
        steps = None
        # one scan per key: a key missing in a row would drop the others there too
        for key in STEP_KEYS:
            if key not in run.summary:
                continue
            part = _scan(run, [key])
            steps = part if steps is None else steps.merge(part, on="_step", how="outer")
        steps = steps if steps is not None else pd.DataFrame(columns=["_step"])
        cache.mkdir(parents=True, exist_ok=True)
        episodes.to_csv(ep_file, index=False)
        steps.to_csv(st_file, index=False)
    # wandb also hands back the rows of the other steps, with every key NaN
    episodes = pd.read_csv(ep_file).dropna(subset=["episode/steps_e"])
    steps = pd.read_csv(st_file)
    steps = steps.dropna(how="all", subset=[k for k in STEP_KEYS if k in steps])
    if ACHIEVED_KEY not in episodes:
        episodes[ACHIEVED_KEY] = np.nan
    return Seed(run.name, episodes, steps)


def load_groups(cache_root: Path, refresh: bool) -> list[Group]:
    api = wandb.Api()
    entity = ENTITY or api.default_entity
    groups = []
    for label, project, wandb_group, names in GROUPS:
        runs = api.runs(f"{entity}/{project}", filters={"group": wandb_group})
        runs = [r for r in runs if names is None or r.name in names]
        print(f"  {label}: {project}/{wandb_group}, {len(runs)} runs")
        if names is not None and len(runs) != len(names):
            missing = names - {r.name for r in runs}
            print(f"    WARNING: not found: {sorted(missing)}")
        group = Group(label)
        for run in sorted(runs, key=lambda r: r.name):
            group.seeds.append(fetch_seed(run, cache_root / project, refresh))
        groups.append(group)
    return groups


def mpc_references(achieved: str) -> dict[str, list[tuple[str, tuple, float]]]:
    """metric key -> [(name, colour, eval mean)] for every MPC reference found."""
    from bachelor.acados_cartpole.plot_benchmark import CONTROLLERS, pick_runs  # imports this module

    wanted = {group for _, group, _, _ in MPC_REFS}
    runs = load_eval_runs([d for d in discover(DEFAULT_ROOTS) if d.name in wanted])
    refs: dict[str, list[tuple[str, tuple, float]]] = {}
    for name, group, dt, color in MPC_REFS:
        controller = next(c for c in CONTROLLERS if c.directory.name == group)
        picked = {path.resolve() for path, _, _ in
                  pick_runs(controller, MPC_N_RUNS, MPC_MIN_DURATION)[0]}
        members = [r for r in runs if r.group == group and r.path.resolve() in picked]
        if not members:
            print(f"  WARNING: no eval runs for {group}")
            continue
        rewards = [r.df["reward"].astype(float) for r in members]
        stab = [r.stabilized_at for r in members]
        if achieved == "censored":
            times = [t if t is not None else MAX_EP_RL_STEPS * RL_DT for t in stab]
        else:
            times = [t for t in stab if t is not None]
        values = {
            # per-sample sum -> per 50 ms, like the `/ N` of the training return
            "episode/cumulative_reward_e": np.mean([rw.sum() * dt / RL_DT for rw in rewards]),
            "episode/stabilized_e": 100.0 * np.mean([t is not None for t in stab]),  # in %
            ACHIEVED_KEY: np.mean(times) if times else np.nan,
            # training logs the cycle MEAN, i.e. the mean per-sample reward
            "step/reward_s": np.mean([rw.mean() for rw in rewards]),
            "step/pi_success_rate_s": np.nanmean(
                [float(r.trailer.get("solver_success_rate", "nan")) for r in members]),
        }
        print(f"  {name}: {len(members)} eval runs, "
              + ", ".join(f"{k.split('/')[-1]}={v:.3g}" for k, v in values.items()))
        for key, value in values.items():
            refs.setdefault(key, []).append((name, color, value))
    return refs


# ALIGNMENT ####################################################################
def seed_series(seed: Seed, metric: Metric, achieved: str) -> tuple[np.ndarray, np.ndarray]:
    """(RL step, value) of one seed, NaN-free, already in plot units."""
    df = seed.episodes if metric.per_episode else seed.steps
    if metric.key not in df:
        return np.array([]), np.array([])
    y = df[metric.key].astype(float)
    if metric.key == ACHIEVED_KEY and achieved == "censored":
        # a failed episode did not stabilize within the episode: count the full length
        failed = df["episode/stabilized_e"].astype(float) == 0
        y = y.where(~failed, MAX_EP_RL_STEPS)
    keep = y.notna().to_numpy()
    return df["_step"].to_numpy(float)[keep], y.to_numpy()[keep] * metric.scale


def align_bin(x: np.ndarray, y: np.ndarray, edges: np.ndarray, metric: Metric) -> np.ndarray:
    """Mean of each window; a running counter keeps its last value instead."""
    out = np.full(edges.size - 1, np.nan)
    if x.size == 0:
        return out
    idx = np.digitize(x, edges) - 1
    for i in range(out.size):
        inside = y[idx == i]
        if inside.size:
            out[i] = inside[-1] if metric.cumulative else inside.mean()
    if metric.cumulative:
        # windows without an episode end: the counter did not change
        s = pd.Series(out).ffill()
        s[: np.searchsorted(edges, x[0], side="right") - 1] = np.nan
        out = s.to_numpy()
    out[edges[:-1] > x[-1]] = np.nan  # this seed had already finished
    return out


def aggregate(group: Group, metric: Metric, args, band: str) -> tuple[np.ndarray, np.ndarray,
                                                            np.ndarray, np.ndarray] | None:
    """(x, mean, band low, band high) across the group's seeds."""
    series = [seed_series(s, metric, args.achieved) for s in group.seeds]
    series = [(x, y) for x, y in series if x.size]
    if not series:
        return None
    end = max(x[-1] for x, _ in series)
    edges = np.arange(0.0, end + args.bin, args.bin)
    centers = 0.5 * (edges[:-1] + edges[1:])
    stack = np.vstack([align_bin(x, y, edges, metric) for x, y in series])
    n_valid = np.sum(~np.isnan(stack), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # columns without any seed
        mean = np.nanmean(stack, axis=0)
        if band == "minmax":
            low, high = np.nanmin(stack, axis=0), np.nanmax(stack, axis=0)
        else:
            std = np.nanstd(stack, axis=0)
            low, high = mean - std, mean + std
    # 0 = all seeds: otherwise the mean jumps every time a seed ends
    need = len(series) if args.min_seeds <= 0 else min(args.min_seeds, len(series))
    for arr in (mean, low, high):
        arr[n_valid < need] = np.nan
    return centers, mean, low, high


# shaded region around the mean: min-max over the seeds, mean +- std
BANDS = ("minmax", "std")


# PLOTTING #####################################################################
def x_values(steps: np.ndarray, unit: str) -> np.ndarray:
    return steps * RL_DT / 60.0 if unit == "minutes" else steps


def x_label(unit: str) -> str:
    if unit == "minutes":
        return "Real-world interaction time [min]"
    return "steps"


def _thousands(value: float, _pos) -> str:
    """10000 -> 10k, 2500 -> 2.5k."""
    if value == 0:
        return "0"
    return f"{value / 1000:g}k" if abs(value) >= 1000 else f"{value:g}"


def draw(ax: plt.Axes, groups: list[Group], metric: Metric, args, refs: dict,
         band: str) -> None:
    for group in groups:
        result = aggregate(group, metric, args, band)
        if result is None:
            continue  # e.g. no MPC in plain SAC -> no solver success rate
        steps, mean, low, high = result
        x = x_values(steps, args.x)
        color = COLORS.get(group.label)
        ax.plot(x, mean, color=color, label=NAMES.get(group.label, group.label))
        ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
    for name, color, value in refs.get(metric.key, []):
        if np.isfinite(value):
            ax.axhline(value, color=color, linestyle="--", linewidth=1.0, label=name)
    title = metric.title
    if metric.key == ACHIEVED_KEY:
        title += " (failures = 10 s)" if args.achieved == "censored" else " (successes only)"
    ax.set_title(title)
    ax.set_ylabel(metric.ylabel)
    ax.set_xlabel(x_label(args.x))
    if args.x == "steps":
        ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    if metric.ylim:
        ax.set_ylim(*metric.ylim)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0)
    ax.set_xlim(left=0)


def save(fig: plt.Figure, out_dir: Path, name: str, formats: tuple[str, ...],
         tight: bool = True) -> None:
    """`tight=False` keeps the figure size, for a figure drawn at its printed size."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        target = out_dir / f"{name}.{fmt}"
        fig.savefig(target, bbox_inches="tight" if tight else None, dpi=300)
        print(f"  wrote {target}")
    plt.close(fig)


def plot_overview(groups: list[Group], refs: dict, args, band: str,
                  formats: tuple[str, ...]) -> None:
    cols = 3
    rows = int(np.ceil(len(METRICS) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows), squeeze=False)
    for ax, metric in zip(axes.flat, METRICS):
        draw(ax, groups, metric, args, refs, band)
    for ax in axes.flat[len(METRICS):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, args.out, f"training_overview_{band}", formats)


def plot_single(groups: list[Group], refs: dict, args, band: str,
                formats: tuple[str, ...]) -> None:
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        draw(ax, groups, metric, args, refs, band)
        ax.legend(fontsize=8, loc="best", frameon=False)
        fig.tight_layout()
        name = metric.key.replace("/", "_").removesuffix("_e").removesuffix("_s")
        save(fig, args.out / "training", f"{name}_{band}", formats)


# figures for the paper: file name -> metrics, one panel each, side by side
PAIRS = {
    "training_return_stabilization": ("episode/stabilized_e", "episode/cumulative_reward_e"),
    "training_violations_solver": ("terminations/total_e", "step/pi_success_rate_s"),
}
# controllers a figure leaves out
PAIR_LEFT_OUT = {"training_return_stabilization": {"real_sac_zop_fullcart"}}
# legend order: the eval table, Pos (not in there) next to its siblings
LEGEND_ORDER = ["MPC-10", "MPC-50", "Angle-10", "Angle", "Cart", "Pos", "SAC"]
# Names in the paper figures (this one and the trajectories of `plot_benchmark`): the
# learned MPC policies as "MPC-SAC (reference, period)", the nominal MPC as
# "MPC (period)", SAC as "SAC (period)".
PAPER_NAMES = {
    "MPC-10": "MPC (10 ms)",
    "MPC-50": "MPC (50 ms)",
    "Angle-10": "MPC-SAC (Angle, 10 ms)",
    "Angle": "MPC-SAC (Angle, 50 ms)",
    "Cart": "MPC-SAC (Cart, 50 ms)",
    "Pos": "MPC-SAC (Pos, 50 ms)",
    "SAC": "SAC (50 ms)",
}


def plot_pairs(groups: list[Group], refs: dict, args, band: str,
               formats: tuple[str, ...], names: tuple[str, ...] = tuple(PAIRS)) -> None:
    """One figure per entry of PAIRS, at the text width of the paper and saved at exactly
    that size (no tight bounding box), so `width=\\textwidth` does not rescale the fonts."""
    by_key = {m.key: m for m in METRICS}
    for name in names:
        keys = PAIRS[name]
        fig, axes = plt.subplots(1, len(keys), figsize=(TEXTWIDTH_IN, 2.2), squeeze=False,
                                 layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04)
        axes = axes.ravel()
        shown = [g for g in groups if g.label not in PAIR_LEFT_OUT.get(name, ())]
        for ax, key in zip(axes, keys):
            draw(ax, shown, by_key[key], args, refs, band)
            ax.set_title("")  # the y label names the metric, the caption the rest
            if key == "terminations/total_e":
                # SAC ends near 70, the SAC-ZOP variants stay below ~15
                ax.set_yscale("symlog", linthresh=1)
                ax.set_ylim(bottom=0)
            if key == "episode/cumulative_reward_e":
                ax.margins(x=0, y=0.02)  # little headroom above the highest band
                ax.autoscale_view()
        # union of both panels: plain SAC has no solver curve, violations no MPC line
        legend = {}
        for ax in axes:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                legend.setdefault(label, handle)
        labels = sorted(legend, key=LEGEND_ORDER.index)
        fig.legend([legend[label] for label in labels],
                   [PAPER_NAMES.get(label, label) for label in labels],
                   # filled by column: MPC | MPC-SAC Angle | the rest
                   loc="outside lower center", ncol=(len(labels) + 1) // 2, frameon=False,
                   handlelength=1.6, handletextpad=0.5, columnspacing=2.0, borderaxespad=0.1)
        save(fig, args.out, f"{name}_{band}", formats, tight=False)


# ENTRY POINT ##################################################################
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bin", type=float, default=500,
                        help="Window width in RL steps (default: 500 = 25 s). "
                             "Keep it above 200 (one full episode), otherwise windows "
                             "without an episode end leave gaps.")
    parser.add_argument("--achieved", choices=["censored", "successes"], default="censored",
                        help="Failed episodes in time-to-stabilize: count as 10 s "
                             "(default) or leave out.")
    parser.add_argument("--min-seeds", type=int, default=0,
                        help="Draw the mean only where at least this many seeds have data "
                             "(default: 0 = all seeds, the curve ends with the shortest seed).")
    parser.add_argument("--x", choices=["minutes", "steps"], default="minutes",
                        help="x axis unit (default: interaction time in minutes).")
    parser.add_argument("--out", type=Path, default=HERE / "paper_plots",
                        help="Output directory (default: paper_plots/ next to this script).")
    parser.add_argument("--cache", type=Path, default=HERE / "paper_plots" / "wandb_cache",
                        help="Where the downloaded wandb history is kept.")
    parser.add_argument("--refresh", action="store_true",
                        help="Download from wandb again instead of using the cache.")
    parser.add_argument("--no-mpc", action="store_true",
                        help="Leave out the MPC eval reference lines.")
    parser.add_argument("--format", nargs="+", default=["pdf"],
                        choices=["png", "pdf", "svg"], help="Output formats.")
    parser.add_argument("--paper", action="store_true",
                        help="Only the figure in the paper, not the other figures.")
    args = parser.parse_args()

    print("loading runs")
    groups = load_groups(args.cache, args.refresh)
    print("\nMPC references")
    refs = {} if args.no_mpc else mpc_references(args.achieved)
    formats = tuple(args.format)
    with paper_font():
        if args.paper:
            plot_pairs(groups, refs, args, PAPER_BAND, formats, (PAPER_PAIR,))
            return
        for band in BANDS:
            print(f"\n{band}: overview + single panels + pairs")
            plot_overview(groups, refs, args, band, formats)
            plot_single(groups, refs, args, band, formats)
            plot_pairs(groups, refs, args, band, formats)


if __name__ == "__main__":
    main()
