"""Benchmark figure: one representative eval trajectory per controller.

Compares the MPC at 10 ms and 50 ms against the SAC-ZOP angle-reference
controller ("full" planner, learnable theta reference) at 10 ms and 50 ms and
the SAC-ZOP cart-reference controller over the cart-only model at 50 ms. For
every controller the ONE run is drawn whose return is closest to the mean
return over all complete runs of that controller, so the figure shows a
typical run rather than the best or worst one.

    python -m bachelor.acados_cartpole.plot_benchmark
    python -m bachelor.acados_cartpole.plot_benchmark --out figures --format png pdf
    python -m bachelor.acados_cartpole.plot_benchmark --pick alt

`--pick alt` is a sensitivity check of that choice: it takes, for the MPC, the
closest run BELOW the mean and, for SAC-ZOP, the closest run ABOVE it, and
writes `benchmark_trajectories_alt.*` next to the default figure.

Two panels on a shared time axis: cumulative reward and theta. Names and colours are the
ones of `plot_training_curves`, so the figure matches the other paper plots.

`--split SECONDS` draws theta only, split in time: the swing-up up to SECONDS
on the left at full scale, the rest on the right with the y-axis zoomed onto
the small oscillations around upright. It writes `benchmark_trajectories[_alt]_zoom.*`.
With `--joined` the two halves are one plot on a continuous time axis, the y-scale
changing at SECONDS, written to `benchmark_trajectories[_alt]_zoom_joined.*`.

    python -m bachelor.acados_cartpole.plot_benchmark --pick alt --split 5 --format pdf png
    python -m bachelor.acados_cartpole.plot_benchmark --pick alt --split 5 --joined --format pdf png

`--all-runs`, with `--split` and `--joined`, draws that joined plot once per controller
of the eval table except Pos, in a 2 x 3 grid (`--ncols 2`: 3 x 2), each with all the runs the table scores
(`pick_runs`) layered instead of one picked run, written to `benchmark_trajectories_zoom_joined_all_runs.*` in the
fonts of the paper (`paper_font`). The angle is reflected at +-pi instead of wound up
(`fold_theta`), so that all panels have the same range and a run spinning for many turns
stays on the axis. The zoomed angle of the controllers in ZOOM_CENTERED is shifted to a
median of 0, against a sensor offset, and samples of a stuck angle sensor (`stuck_theta`)
are left out. The table keeps them.

    python -m bachelor.acados_cartpole.plot_benchmark --split 5 --joined --all-runs --format pdf png

The reward is the logged `reward` column, i.e. the learning reward
(`compute_reward` -> HARDWARE_REWARD in my_helpers.py). The return is its sum
scaled by `dt / 50 ms`, as in `plot_eval_runs`, so 10 ms and 50 ms runs with
the same behaviour get the same number. The cumulative-reward panel uses the
same scaling, so each curve ends at its run's return. Tripped or truncated runs are dropped
(see `plot_eval_runs.is_complete`) and neither count for the mean nor get
picked. Only the first N_RUNS complete runs in run order are used. `eval_table` and
`--all-runs` select theirs with `pick_runs` instead.

Theta is unwrapped, so a swing over the +-pi seam is drawn as one continuous
line, and then shifted by a multiple of 2 pi so the run ENDS near 0: every
controller balances at the same height of the panel, whatever direction it
swung up in.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FuncFormatter, MultipleLocator

from bachelor.acados_cartpole.plot_eval_runs import (
    DEFAULT_CUM_REF_DT,
    UPRIGHT_THRESHOLD,
    is_complete,
    read_trailer,
    run_dt,
    run_index,
)
from bachelor.acados_cartpole.plot_training_curves import (
    COLORS,
    MPC_REFS,
    NAMES,
    PAPER_NAMES,
    TEXTWIDTH_IN,
    paper_font,
)

HERE = Path(__file__).resolve().parent

# A sample period longer than this many median periods is a gap in the log and
# is left blank in the plot instead of being bridged by a straight line.
GAP_FACTOR = 5.0

# `pick_runs` does not use a run missing more samples than this in one gap of its log.
MAX_GAP_SAMPLES = 10

# Headroom of the zoomed theta panel above the largest |theta| it shows.
ZOOM_MARGIN = 1.2

# Half-height (rad) of the zoomed theta axis of `--all-runs`, and its ticks: the upright band
# (UPRIGHT_THRESHOLD = 0.15 rad) fills its middle half.
ALL_RUNS_ZOOM = 0.3
ALL_RUNS_ZOOM_TICKS = (-0.15, 0.0, 0.15)

# Complete runs per controller, the first ones in run order; the same runs
# `eval_table` scores.
N_RUNS = 20

EPISODE_S = 10.0

# Controller label -> its name in the eval table, where it differs.
TABLE_NAMES = {"MPC (10 ms)": "MPC-10", "MPC (50 ms)": "MPC-50"}

# Controllers whose balancing sits off 0 by a sensor offset: `plot_theta_joined_runs`
# shifts their zoomed angle so that its median after the split is 0.
ZOOM_CENTERED = {NAMES["real_sac_zop_cart"]}

# Controllers of the eval table that `--all-runs` does not draw.
ALL_RUNS_LEFT_OUT = {NAMES["real_sac_zop_fullcart"]}

# The value the angle sensor sticks at for some samples, whatever the pole does, and the
# step in one sample (rad) that tells such a block from the pole passing that angle.
STUCK_THETA = -1.02
STUCK_STEP = 0.3

_MPC_COLORS ={directory: color for _, directory, _, color in MPC_REFS}


@dataclass(frozen=True)
class Controller:
    label: str
    directory: Path
    color: tuple
    alt_side: int  # side of the mean `--pick alt` takes the run from: -1 below, +1 above
    alt_rank: int = 0  # with `--pick alt`: 0 = closest run on that side, 1 = second closest, ...
    dt: float = 0.05  # nominal control period (s)
    spare_runs: frozenset[int] = frozenset()  # run numbers `pick_runs` only uses for a gap


CONTROLLERS: tuple[Controller, ...] = (
    Controller("MPC (10 ms)", HERE / "real_acados/eval_logs/mpc_full_10ms_saz",
               _MPC_COLORS["mpc_full_10ms_saz"], -1, dt=0.01,
               spare_runs=frozenset({8, 9, 11, 13, 19})),
    Controller("MPC (50 ms)", HERE / "real_acados/eval_logs/mpc_full_50ms_saz",
               _MPC_COLORS["mpc_full_50ms_saz"], -1),
    Controller(NAMES["real_sac_zop_full_10ms"],
               HERE / "real_sac_zop/checkpoints/real_sac_zop_full_10ms/s4",
               COLORS["real_sac_zop_full_10ms"], +1, dt=0.01),
    Controller(NAMES["real_sac_zop_full"], HERE / "real_sac_zop/checkpoints/real_sac_zop_full/s9",
               COLORS["real_sac_zop_full"], +1, spare_runs=frozenset({2, 5, 10, 13, 14})),
    Controller(NAMES["real_sac_zop_cart"], HERE / "real_sac_zop/checkpoints/real_sac_zop_cart",
               COLORS["real_sac_zop_cart"], +1),
)

# The benchmark controllers plus Pos and plain SAC, which the benchmark figure does
# not show.
TABLE_CONTROLLERS = (
    *CONTROLLERS[:4],
    Controller(NAMES["real_sac_zop_fullcart"],
               HERE / "real_sac_zop/checkpoints/real_sac_zop_fullcart/s1",
               COLORS["real_sac_zop_fullcart"], +1),
    CONTROLLERS[4],
    Controller(NAMES["real_sac"], HERE / "real_sac_zop/checkpoints/real_sac/seed4",
               COLORS["real_sac"], +1),
)


@dataclass
class Run:
    path: Path
    df: pd.DataFrame
    ret: float  # return per 50 ms
    scale: float  # dt / 50 ms, turns the per-sample running sum into "per 50 ms"


def load_runs(directory: Path, min_duration: float, n_runs: int = N_RUNS) -> list[Run]:
    """The first `n_runs` complete eval runs below `directory` in run order, with
    their rate-normalized return."""
    runs = []
    files = sorted(
        (p for p in directory.rglob("eval_*.csv") if p.name != "eval_summary.csv"),
        key=lambda p: (run_index(p), str(p)),
    )
    for path in files:
        df = pd.read_csv(path, comment="#")
        if df.empty or "reward" not in df:
            continue
        complete, _ = is_complete(df, read_trailer(path), min_duration)
        if not complete:
            continue
        scale = run_dt(df) / DEFAULT_CUM_REF_DT
        runs.append(Run(path, df, float(df["reward"].astype(float).sum()) * scale, scale))
        if len(runs) == n_runs:
            break
    return runs


def missing_samples(df: pd.DataFrame, dt: float) -> float:
    """The samples missing in the largest gap of a log, at the nominal period `dt`."""
    return float(np.diff(df["time_s"].to_numpy(dtype=float)).max(initial=dt)) / dt - 1.0


def pick_runs(c: Controller, n_runs: int, min_duration: float):
    """The first `n_runs` finished runs of a controller in run order, topped up with the
    earliest failed runs if there are not enough. A run with more than MAX_GAP_SAMPLES in
    a gap is never used. A clean run of `c.spare_runs` takes its place, in run order, and
    the spare runs are left out otherwise. Returns the picked (path, frame, reason)
    triples, reason None for a finished run, the failed runs among the first `n_runs`
    that got replaced, as (path, reason), and the paths of the runs dropped for a gap."""
    def order(path: Path) -> tuple[int, str]:
        return run_index(path), str(path)

    runs, spares, gapped = [], [], []
    for path in sorted((p for p in c.directory.rglob("eval_*.csv") if p.name != "eval_summary.csv"),
                       key=order):
        df = pd.read_csv(path, comment="#")
        complete, reason = is_complete(df, read_trailer(path), min_duration)
        has_gap = complete and missing_samples(df, c.dt) > MAX_GAP_SAMPLES
        if run_index(path) in c.spare_runs:
            if complete and not has_gap:
                spares.append((path, df, None))
        elif has_gap:
            gapped.append(path)
        else:
            runs.append((path, df, None if complete else reason))
    done = [run for run in runs if run[2] is None]
    failed = [run for run in runs if run[2] is not None]
    stand_ins = spares[:len(gapped)]
    picked = sorted(done[:n_runs - len(stand_ins)] + stand_ins, key=lambda run: order(run[0]))
    picked += failed[:max(0, n_runs - len(picked))]
    picked_paths = {path for path, _, _ in picked}
    replaced = [(path, reason) for path, _, reason in runs[:n_runs]
                if reason is not None and path not in picked_paths]
    return picked, replaced, gapped


def representative(runs: list[Run], side: int = 0, rank: int = 0) -> tuple[Run, float]:
    """The run whose return is closest to the mean return, and that mean.

    `side` restricts the choice to runs below (-1) or above (+1) the mean; it
    falls back to all runs if that side is empty. `rank` takes the rank-th
    closest instead (0 = closest), clamped to the runs available.
    """
    mean = float(np.mean([r.ret for r in runs]))
    pool = [r for r in runs if side == 0 or np.sign(r.ret - mean) == side] or runs
    pool = sorted(pool, key=lambda r: abs(r.ret - mean))
    return pool[min(rank, len(pool) - 1)], mean


def continuous_theta(theta: np.ndarray) -> np.ndarray:
    """Unwrapped angle, shifted by k * 2 pi so the run ends near upright = 0."""
    unwrapped = np.unwrap(theta.astype(float))
    return unwrapped - 2 * np.pi * np.round(unwrapped[-1] / (2 * np.pi))


def stuck_theta(raw: np.ndarray) -> np.ndarray:
    """Mask of the samples of a logged angle where the sensor is stuck at STUCK_THETA:
    blocks at that value entered or left by a step larger than STUCK_STEP."""
    at = np.abs(raw - STUCK_THETA) < 0.006
    step = np.abs((np.diff(raw) + np.pi) % (2 * np.pi) - np.pi)
    mask = np.zeros(len(raw), dtype=bool)
    for start, stop in np.flatnonzero(np.diff(np.r_[False, at, False])).reshape(-1, 2):
        enter = step[start - 1] if start > 0 else 0.0
        leave = step[stop - 1] if stop < len(raw) else 0.0
        mask[start:stop] = max(enter, leave) > STUCK_STEP
    return mask


def fold_theta(theta: np.ndarray) -> np.ndarray:
    """A `continuous_theta` reflected at +-pi (hanging) instead of wound up.

    Stays continuous, starts at +-pi whatever the run does later, and every pass through
    0 is a pass through upright. The absolute value is the true distance from upright;
    the sign is that of theta only after the last turn, so where a run has settled.
    """
    return 2 * np.arcsin(np.sin(theta / 2))


def with_gaps(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN at gaps in the log (e.g. a serial dropout), so no line bridges them."""
    t, y = t.astype(float), y.astype(float)
    dt = np.diff(t)
    idx = np.flatnonzero(dt > GAP_FACTOR * np.median(dt)) + 1
    return np.insert(t, idx, np.nan), np.insert(y, idx, np.nan)


def pi_formatter(value: float, _pos) -> str:
    k = int(round(value / np.pi))
    return {0: "$0$", 1: r"$\pi$", -1: r"$-\pi$"}.get(k, rf"${k}\pi$")


def trajectory(run: Run) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = run.df["time_s"].to_numpy(dtype=float)
    theta = continuous_theta(run.df["theta_rad"].to_numpy(dtype=float))
    cum_reward = run.df["reward"].astype(float).cumsum().to_numpy() * run.scale
    return t, theta, cum_reward


def style_theta_full(ax) -> None:
    ax.set_ylabel(r"$\theta$ [rad]")
    ax.axhspan(-UPRIGHT_THRESHOLD, UPRIGHT_THRESHOLD, color="g", alpha=0.12, linewidth=0)
    ax.yaxis.set_major_locator(MultipleLocator(np.pi))
    ax.yaxis.set_major_formatter(FuncFormatter(pi_formatter))
    ax.set_xlabel("Time [s]")


def plot_reward_theta(picked: list[tuple[Controller, Run]]) -> plt.Figure:
    fig, (ax_r, ax_th) = plt.subplots(2, 1, sharex=True, figsize=(9, 4.8))
    for c, run in picked:
        t, theta, cum_reward = trajectory(run)
        ax_r.plot(*with_gaps(t, cum_reward), color=c.color, linewidth=1.6, label=c.label)
        ax_th.plot(*with_gaps(t, theta), color=c.color, linewidth=1.6)

    ax_r.set_ylabel(f"cum. reward (per {DEFAULT_CUM_REF_DT * 1000:.0f} ms)")
    ax_r.set_ylim(bottom=0)
    style_theta_full(ax_th)
    for ax in (ax_r, ax_th):
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)

    ax_r.set_title("Representative eval run per controller", fontsize=11)
    fig.legend(*ax_r.get_legend_handles_labels(), loc="lower center", ncol=len(picked),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def theta_curves(picked: list[tuple[Controller, Run]], split: float):
    """(controller, t, theta) per run, the last time stamp, and the half-height of the
    zoomed theta range: the largest |theta| after `split`, at least the upright band."""
    curves = [(c, *trajectory(run)[:2]) for c, run in picked]
    t_end = max(float(t[-1]) for _, t, _ in curves)
    peak = max(float(np.abs(theta[t >= split]).max()) for _, t, theta in curves)
    return curves, t_end, ZOOM_MARGIN * max(UPRIGHT_THRESHOLD, peak)


def plot_theta_split(picked: list[tuple[Controller, Run]], split: float) -> plt.Figure:
    """Theta only: swing-up until `split` at full scale, the rest zoomed in y."""
    fig, (ax_l, ax_z) = plt.subplots(1, 2, figsize=(9, 3.4))
    curves, t_end, zoom = theta_curves(picked, split)
    for c, t, theta in curves:
        for ax in (ax_l, ax_z):
            ax.plot(*with_gaps(t, theta), color=c.color, linewidth=1.6, label=c.label)

    style_theta_full(ax_l)
    ax_l.set_xlim(0, split)
    ax_l.set_title("Swing-up", fontsize=10)

    ax_z.axhspan(-UPRIGHT_THRESHOLD, UPRIGHT_THRESHOLD, color="g", alpha=0.12, linewidth=0)
    ax_z.set_xlim(split, t_end)
    ax_z.set_ylim(-zoom, zoom)
    ax_z.yaxis.set_major_locator(MultipleLocator(0.05))
    ax_z.yaxis.tick_right()
    ax_z.yaxis.set_label_position("right")
    ax_z.set_ylabel(r"$\theta$ [rad]")
    ax_z.set_xlabel("Time [s]")
    ax_z.set_title(r"balancing ($\theta$ zoomed)", fontsize=10)
    for ax in (ax_l, ax_z):
        ax.grid(True, alpha=0.3)

    fig.suptitle("Representative eval run per controller", fontsize=11)
    fig.legend(*ax_l.get_legend_handles_labels(), loc="lower center", ncol=len(picked),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=4)

    # Funnel from the zoomed band at the edge of the swing-up panel to the zoom panel,
    # added after tight_layout, which cannot place figure-level patches.
    for y_l, y_z in ((zoom, 1), (-zoom, 0)):
        fig.add_artist(ConnectionPatch(
            xyA=(split, y_l), coordsA=ax_l.transData, xyB=(0, y_z), coordsB=ax_z.transAxes,
            color="0.5", linewidth=0.8, linestyle="--"))
    return fig


def join_theta_axes(ax_l, ax_z, split: float, t_end: float, zoom: float) -> None:
    """Glue a swing-up axis, its y-range already set, and a zoomed axis into one plot.

    The zoomed axis has its own y-axis on the right, and its range is placed so
    theta = 0 sits at the same height on both sides: a balanced pendulum runs
    straight through `split`.
    """
    ax_l.set_xlim(0, split)
    lo, hi = ax_l.get_ylim()
    zero = -lo / (hi - lo)  # height of theta = 0 as a fraction of the axis
    span = zoom / min(zero, 1 - zero)
    ax_z.axhspan(-UPRIGHT_THRESHOLD, UPRIGHT_THRESHOLD, color="g", alpha=0.12, linewidth=0)
    ax_z.set_xlim(split, t_end)
    ax_z.set_ylim(-zero * span, (1 - zero) * span)
    ax_z.yaxis.tick_right()
    ax_z.yaxis.set_label_position("right")

    # One time axis: same tick spacing on both sides, `split` labelled once, and a
    # dashed line instead of the two touching spines where the y-scale changes.
    for ax in (ax_l, ax_z):
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.grid(True, alpha=0.3)
    ax_z.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _pos: "" if np.isclose(v, split) else f"${v:g}$"))
    ax_l.spines["right"].set_visible(False)
    ax_z.spines["left"].set_visible(False)
    ax_z.axvline(split, color="0.4", linewidth=1.0, linestyle="--", clip_on=False)


def plot_theta_joined(picked: list[tuple[Controller, Run]], split: float) -> plt.Figure:
    """Theta only on one continuous time axis, full scale until `split`, zoomed in y after.

    Two axes glued together without a gap, with widths proportional to their time
    spans, so seconds per cm are the same on both sides.
    """
    curves, t_end, zoom = theta_curves(picked, split)
    fig, (ax_l, ax_z) = plt.subplots(1, 2, figsize=(9, 3.4),
                                     gridspec_kw={"width_ratios": [split, t_end - split]})
    for c, t, theta in curves:
        for ax in (ax_l, ax_z):
            ax.plot(*with_gaps(t, theta), color=c.color, linewidth=1.6, label=c.label)

    style_theta_full(ax_l)
    ax_l.xaxis.label.set_x(0.5 * t_end / split)  # centre "Time [s]" under the whole axis
    ax_l.set_title("Swing-up", fontsize=10)

    join_theta_axes(ax_l, ax_z, split, t_end, zoom)
    ax_z.yaxis.set_major_locator(MultipleLocator(0.05))
    ax_z.set_ylabel(r"$\theta$ [rad], zoomed")
    ax_z.set_title(r"balancing ($\theta$ zoomed)", fontsize=10)

    fig.suptitle("Representative eval run per controller", fontsize=11)
    fig.legend(*ax_l.get_legend_handles_labels(), loc="lower center", ncol=len(picked),
               fontsize=9, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.subplots_adjust(wspace=0)  # after tight_layout, which would reopen the gap
    return fig


def plot_theta_joined_runs(groups: list[tuple[Controller, list]], split: float,
                           ncols: int = 3, height_scale: float = 1.0) -> plt.Figure:
    """`plot_theta_joined` per controller in a grid of `ncols` columns, all runs layered.

    `groups` holds the `pick_runs` triples per controller. Drawn at the text width of the
    paper and saved at that size, so the fonts are true to size (`paper_font`). Theta goes
    through `fold_theta`, so all panels share one range, and the zoomed range does not grow
    with the runs either: a run that has not settled by `split` leaves the zoomed axis and
    returns once it has. Since all panels share both ranges, only the outer columns carry
    the y tick labels and only the bottom row the time labels. `height_scale` scales the
    height of the panels, not the margins. Names are the PAPER_NAMES of the training
    figure.
    """
    zoom = ALL_RUNS_ZOOM
    nrows = -(-len(groups) // ncols)
    # layout in inches: margins, gap between the cells, header line, name line
    left, right, gap, bottom = 0.40, 0.56, 0.20, 0.36
    header, title, row_gap = 0.15, 0.17, 0.12
    width = TEXTWIDTH_IN
    cell_w = (width - left - right - (ncols - 1) * gap) / ncols
    cell_h = 1.12 * height_scale
    height = bottom + nrows * (cell_h + title) + (nrows - 1) * row_gap + header
    fig = plt.figure(figsize=(width, height))
    w_l = cell_w * split / EPISODE_S

    axes = []
    for i, (c, picked) in enumerate(groups):
        row, col = divmod(i, ncols)
        x0 = left + col * (cell_w + gap)
        y0 = height - header - title - row * (cell_h + title + row_gap) - cell_h
        ax_l = fig.add_axes((x0 / width, y0 / height, w_l / width, cell_h / height))
        ax_z = fig.add_axes(((x0 + w_l) / width, y0 / height, (cell_w - w_l) / width,
                             cell_h / height))
        axes.append((ax_l, ax_z))

        name = TABLE_NAMES.get(c.label, c.label)
        name = PAPER_NAMES.get(name, name)
        curves, n_stuck = [], 0
        for _, df, _ in picked:
            df = df[df["time_s"] <= EPISODE_S]  # as `eval_table` scores it
            raw = df["theta_rad"].to_numpy(dtype=float)
            stuck = stuck_theta(raw)
            n_stuck += int(stuck.sum())
            t = df["time_s"].to_numpy(dtype=float)[~stuck]
            curves.append((t, fold_theta(continuous_theta(raw[~stuck]))))
        if n_stuck:
            print(f"{name}: left out {n_stuck} samples of a stuck angle sensor")

        shift = 0.0
        if c.label in ZOOM_CENTERED:
            shift = float(np.median(np.concatenate([y[t >= split] for t, y in curves])))
            print(f"{name}: zoomed theta shifted by {-shift:+.4f} rad")
        for t, y in curves:
            ax_l.plot(*with_gaps(t, y), color=c.color, linewidth=0.5, alpha=0.55)
            ax_z.plot(*with_gaps(t, y - shift), color=c.color, linewidth=0.5, alpha=0.55)

        style_theta_full(ax_l)
        ax_l.set(xlabel="", ylabel="", ylim=(-1.12 * np.pi, 1.12 * np.pi))
        join_theta_axes(ax_l, ax_z, split, EPISODE_S, zoom)
        ax_z.set_yticks(ALL_RUNS_ZOOM_TICKS)
        ax_l.set_title(name, loc="left")
        for ax in (ax_l, ax_z):
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.grid(True, which="both", alpha=0.3)
        if col > 0:
            ax_l.tick_params(labelleft=False)
        if col < ncols - 1 and i < len(groups) - 1:
            ax_z.tick_params(labelright=False)
        if i + ncols < len(groups):  # another panel below
            ax_l.tick_params(labelbottom=False)
            ax_z.tick_params(labelbottom=False)
        if row == 0:
            for ax, phase in ((ax_l, "Swing-up"), (ax_z, "Balancing")):
                box = ax.get_position()
                # 2 pt below the controller names, so they read as a side note
                fig.text(box.x0 + box.width / 2, 1 - 0.02 / height, phase, ha="center",
                         va="top", fontsize=plt.rcParams["axes.titlesize"] - 2,
                         style="italic", color="0.4")
        if i + ncols >= len(groups):  # lowest panel of its column
            ax_l.set_xlabel("Time [s]")
            ax_l.xaxis.label.set_x(0.5 * EPISODE_S / split)  # centre under the whole cell

    fig.text(0.04 / width, 0.5, r"$\theta$ [rad]", rotation=90,
             va="center", ha="left")
    fig.text(1 - 0.04 / width, 0.5, r"$\theta$ [rad], zoomed", rotation=90, va="center",
             ha="right")
    return fig


def save_figure(fig: plt.Figure, args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = ("" if args.pick == "nearest" else "_alt") + ("" if args.split is None else "_zoom")
    suffix += "_joined" if args.joined else ""
    suffix += "_all_runs" if args.all_runs else ""
    suffix += f"_{-(-6 // args.ncols)}x{args.ncols}" if args.all_runs and args.ncols != 3 else ""
    suffix += "_compact" if args.all_runs and args.height_scale != 1.0 else ""
    for fmt in args.format:
        target = args.out / f"benchmark_trajectories{suffix}.{fmt}"
        # a figure drawn at its printed size keeps that size
        fig.savefig(target, bbox_inches=None if args.all_runs else "tight", dpi=300)
        print(f"wrote {target}")
    if args.show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=HERE / "paper_plots",
                        help="Output directory (default: paper_plots/ next to this script).")
    parser.add_argument("--format", nargs="+", default=["pdf"],
                        choices=["png", "pdf", "svg"], help="Output formats.")
    parser.add_argument("--min-duration", type=float, default=9.5,
                        help="Drop runs shorter than this many seconds (default: 9.5).")
    parser.add_argument("--runs", type=int, default=N_RUNS,
                        help=f"Complete runs per controller (default: {N_RUNS}).")
    parser.add_argument("--pick", choices=["nearest", "alt"], default="nearest",
                        help="nearest: run closest to the mean (default). alt: closest "
                             "run below the mean for the MPC, above it for SAC-ZOP.")
    parser.add_argument("--split", type=float, default=None, metavar="SECONDS",
                        help="Theta only, split at SECONDS: swing-up at full scale on the "
                             "left, the rest with a zoomed y-axis on the right.")
    parser.add_argument("--joined", action="store_true",
                        help="With --split: one plot on a continuous time axis instead of "
                             "two panels.")
    parser.add_argument("--ncols", type=int, default=3,
                        help="With --all-runs: panels per row (default: 3, i.e. 2 x 3). "
                             "Other values add the grid to the file name, e.g. _3x2.")
    parser.add_argument("--height-scale", type=float, default=1.0,
                        help="With --all-runs: scale the panel height, e.g. 0.75; adds "
                             "_compact to the file name.")
    parser.add_argument("--all-runs", action="store_true",
                        help="With --split and --joined: one row per controller of the eval "
                             "table, with all the runs it scores.")
    parser.add_argument("--show", action="store_true", help="Show the figure.")
    args = parser.parse_args()
    if args.joined and args.split is None:
        parser.error("--joined needs --split")
    if args.all_runs and not args.joined:
        parser.error("--all-runs needs --joined")

    if args.all_runs:
        groups = []
        for c in TABLE_CONTROLLERS:
            if c.label in ALL_RUNS_LEFT_OUT:
                continue
            picked, *_ = pick_runs(c, args.runs, args.min_duration)
            print(f"{c.label}: " + ", ".join(path.stem for path, _, _ in picked))
            groups.append((c, picked))
        # also around the save: usetex and its preamble are read when the text is drawn
        with paper_font():
            save_figure(plot_theta_joined_runs(groups, args.split, args.ncols, args.height_scale), args)
    else:
        picked = []
        print(f"{'controller':<24}{'runs':>5}{'mean return':>13}{'picked':>9}  run")
        for c in CONTROLLERS:
            runs = load_runs(c.directory, args.min_duration, args.runs)
            if not runs:
                print(f"{c.label:<24}  no complete runs in {c.directory}")
                continue
            if args.pick == "alt":
                run, mean = representative(runs, c.alt_side, c.alt_rank)
            else:
                run, mean = representative(runs)
            print(f"{c.label:<24}{len(runs):>5}{mean:>13.1f}{run.ret:>9.1f}  "
                  f"{run.path.relative_to(c.directory)}")
            picked.append((c, run))

        if args.split is None:
            fig = plot_reward_theta(picked)
        elif args.joined:
            fig = plot_theta_joined(picked, args.split)
        else:
            fig = plot_theta_split(picked, args.split)
        save_figure(fig, args)


if __name__ == "__main__":
    main()
