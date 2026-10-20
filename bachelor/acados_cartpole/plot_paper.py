"""Performance-comparison figures for the paper.

Reads the same eval CSVs as `plot_eval_runs` (SAC-ZOP checkpoints and MPC eval
logs) plus the per-seed training logs (`meta.pth`) and writes two figures:

1. `stabilization_ecdf`   - fraction of eval runs stabilized by time t, one
   curve per controller, split into 10 ms and 50 ms. Failed and tripped runs
   stay in the denominator, so a curve that ends at 0.75 means a quarter of the
   runs never got up.
2. `learning_curves`      - training episode return against real interaction
   time, one thin line per seed and the mean across seeds; the MPC eval returns
   are drawn as horizontal references.

    python -m bachelor.acados_cartpole.plot_paper
    python -m bachelor.acados_cartpole.plot_paper --out figures --format pdf png

Eval returns are per-sample sums, so they are scaled by `dt / 50 ms` like the
cumulative panels of `plot_eval_runs` - a 10 ms run and a 50 ms run with the
same behaviour then get the same number. The training returns in `meta.pth` are
already per RL decision (50 ms in every variant, the 10 ms variants hold the
parameter for 5 MPC steps), so they need no scaling.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from bachelor.acados_cartpole.plot_eval_runs import (
    DEFAULT_CUM_REF_DT,
    DEFAULT_ROOTS,
    RATE_SPLIT_DT,
    discover,
    read_trailer,
    run_dt,
    run_index,
)

HERE = Path(__file__).resolve().parent
SAC_ZOP_ROOT = HERE / "real_sac_zop" / "checkpoints"

# Every variant's RL decision period. meta.pth counts RL steps, not MPC steps.
RL_DT = 0.05


@dataclass
class EvalRun:
    group: str
    path: Path
    df: pd.DataFrame
    trailer: dict[str, str]
    dt: float

    @property
    def scale(self) -> float:
        """Factor that expresses this run's per-sample sums per 50 ms."""
        return self.dt / DEFAULT_CUM_REF_DT if np.isfinite(self.dt) else 1.0

    @property
    def rate(self) -> str:
        return "10ms" if self.dt < RATE_SPLIT_DT else "50ms"

    @property
    def ret(self) -> float:
        try:
            value = float(self.trailer["episodic_return"])
        except (KeyError, ValueError):
            value = float(self.df["reward"].astype(float).sum())
        return value * self.scale

    @property
    def stabilized_at(self) -> float | None:
        """Time (s) the run counted as stabilized, None if it never did."""
        if _float(self.trailer.get("stabilized")) == 0:
            return None
        t = _float(self.trailer.get("stabilized_at_s"))
        if t is not None:
            return t
        step = _float(self.trailer.get("stabilized_at_step"))
        if step is None:
            return None
        if "step" in self.df:
            hit = self.df.loc[self.df["step"].astype(float) >= step, "time_s"]
            if not hit.empty:
                return float(hit.iloc[0])
        return step * self.dt


def _float(value: str | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# READING ######################################################################
def load_eval_runs(directories: list[Path]) -> list[EvalRun]:
    """Every eval run of every group, tripped ones included."""
    runs = []
    for directory in directories:
        files = sorted(
            (p for p in directory.rglob("eval_*.csv") if p.name != "eval_summary.csv"),
            key=lambda p: (run_index(p), str(p)),
        )
        for path in files:
            try:
                df = pd.read_csv(path, comment="#")
            except Exception as e:
                print(f"  skipped {path}: unreadable ({e})")
                continue
            if df.empty or "time_s" not in df:
                continue
            runs.append(EvalRun(directory.name, path, df, read_trailer(path), run_dt(df)))
    return runs


def load_training(root: Path) -> dict[str, list[pd.DataFrame]]:
    """Per-seed training episodes from every `meta.pth` below `root`.

    A `meta.pth` directly in a group directory is usually a copy of one of its
    seeds, so seeds are deduplicated by their final step and episode count.
    """
    curves: dict[str, list[pd.DataFrame]] = {}
    seen: set[tuple[str, int, int]] = set()
    for path in sorted(root.glob("*/**/meta.pth")):
        group = path.relative_to(root).parts[0]
        try:
            meta = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  skipped {path}: unreadable ({e})")
            continue
        key = (group, int(meta.get("abs_step_count", -1)), int(meta.get("episode_count", -1)))
        episodes = meta.get("episode_rewards", [])
        if key in seen or not episodes:
            continue
        seen.add(key)
        df = pd.DataFrame(episodes)
        df["time_min"] = df["steps"].cumsum() * RL_DT / 60.0
        curves.setdefault(group, []).append(df)
    return curves


def color_map(names: list[str]) -> dict[str, tuple]:
    """One fixed colour per controller, shared by every figure."""
    names = sorted(set(names))
    colors = plt.cm.tab10(np.arange(len(names)) % 10)
    return {name: tuple(c) for name, c in zip(names, colors)}


def save(fig: plt.Figure, out_dir: Path, name: str, formats: tuple[str, ...]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        target = out_dir / f"{name}.{fmt}"
        fig.savefig(target, bbox_inches="tight", dpi=200)
        print(f"  wrote {target}")
    plt.close(fig)


# 1. STABILIZATION ECDF ########################################################
def plot_stabilization_ecdf(runs: list[EvalRun], colors: dict, out_dir: Path,
                            formats: tuple[str, ...]) -> None:
    rates = [r for r in ("10ms", "50ms") if any(run.rate == r for run in runs)]
    fig, axes = plt.subplots(1, len(rates), figsize=(5.5 * len(rates), 4), sharey=True,
                             squeeze=False)
    for ax, rate in zip(axes[0], rates):
        groups = sorted({r.group for r in runs if r.rate == rate})
        t_end = max(float(r.df["time_s"].iloc[-1]) for r in runs if r.rate == rate)
        for group in groups:
            members = [r for r in runs if r.group == group]
            times = np.sort([t for t in (r.stabilized_at for r in members) if t is not None])
            n = len(members)
            # step function: fraction of ALL runs stabilized by time t
            t = np.concatenate([[0.0], times, [t_end]])
            frac = np.concatenate([[0.0], np.arange(1, times.size + 1) / n,
                                   [times.size / n]])
            median = f", median {np.median(times):.1f} s" if times.size else ""
            ax.step(t, frac, where="post", color=colors[group], linewidth=2.0,
                    label=f"{group}: {times.size}/{n}{median}")
        ax.set_title(f"{rate[:-2]} ms control", fontsize=11, fontweight="bold")
        ax.set_xlabel("time (s)")
        ax.set_xlim(0, t_end)
        ax.set_ylim(0, 1.03)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    axes[0, 0].set_ylabel("fraction of eval runs stabilized")
    fig.suptitle("Time to stabilization (all eval runs, failures included)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, out_dir, "stabilization_ecdf", formats)


# 2. LEARNING CURVES ###########################################################
def plot_learning_curves(curves: dict[str, list[pd.DataFrame]], runs: list[EvalRun],
                         colors: dict, out_dir: Path, formats: tuple[str, ...],
                         smooth: int, min_seeds: int) -> None:
    if not curves:
        print("  no meta.pth found - learning curves skipped")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for group in sorted(curves):
        seeds = curves[group]
        end = max(float(s["time_min"].iloc[-1]) for s in seeds)
        grid = np.linspace(0.0, end, 400)
        stack = []
        for s in seeds:
            y = s["cumulative_reward"].rolling(smooth, min_periods=1).mean().to_numpy()
            t = s["time_min"].to_numpy()
            ax.plot(t, y, color=colors[group], linewidth=0.6, alpha=0.3)
            row = np.interp(grid, t, y)
            row[grid > t[-1]] = np.nan  # a finished seed does not pull the mean
            stack.append(row)
        stack = np.vstack(stack)
        n_valid = np.sum(~np.isnan(stack), axis=0)
        with np.errstate(all="ignore"):
            mean = np.nanmean(stack, axis=0)
        # 0 = all seeds: otherwise the mean jumps every time a seed ends
        need = len(seeds) if min_seeds <= 0 else min(min_seeds, len(seeds))
        mean[n_valid < need] = np.nan
        ax.plot(grid, mean, color=colors[group], linewidth=2.2,
                label=f"{group} ({len(seeds)} seeds)")

    # MPC needs no training: its eval return is the bar to reach
    for group in sorted({r.group for r in runs if r.group.startswith("mpc")}):
        rets = [r.ret for r in runs if r.group == group]
        ax.axhline(np.mean(rets), color=colors[group], linestyle="--", linewidth=1.4,
                   label=f"{group} eval mean (no training)")

    ax.set_xlabel("real-world interaction time (min)")
    ax.set_ylabel(f"training episode return (per 50 ms, {smooth}-episode mean)")
    ax.set_title("Learning on the real system: thin = seeds, bold = mean across seeds",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.margins(x=0)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, out_dir, "learning_curves", formats)


# ENTRY POINT ##################################################################
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dirs", nargs="+", type=Path,
                        help="Eval directories. Default: everything `plot_eval_runs` finds.")
    parser.add_argument("--out", type=Path, default=HERE / "paper_plots",
                        help="Output directory (default: paper_plots/ next to this script).")
    parser.add_argument("--format", nargs="+", default=["png"],
                        choices=["png", "pdf", "svg"], help="Output formats.")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Episodes in the learning-curve moving average (default: 5).")
    parser.add_argument("--min-seeds", type=int, default=0,
                        help="Draw the learning-curve mean only where at least this "
                             "many seeds are still running (default: 0 = all seeds, "
                             "i.e. the mean ends with the shortest seed).")
    args = parser.parse_args()

    directories = args.dirs if args.dirs else discover(DEFAULT_ROOTS)
    runs = load_eval_runs(directories)
    curves = load_training(SAC_ZOP_ROOT)
    if not runs and not curves:
        raise SystemExit("No eval CSVs or meta.pth found.")
    print(f"{len(runs)} eval runs in {len({r.group for r in runs})} groups, "
          f"training logs for {len(curves)} groups")

    colors = color_map([r.group for r in runs] + list(curves))
    formats = tuple(args.format)

    print("\n1. stabilization ECDF")
    plot_stabilization_ecdf(runs, colors, args.out, formats)
    print("\n2. learning curves")
    plot_learning_curves(curves, runs, colors, args.out, formats, args.smooth,
                         args.min_seeds)


if __name__ == "__main__":
    main()
