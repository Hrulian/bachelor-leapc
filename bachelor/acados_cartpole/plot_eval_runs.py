"""Stacked overview plot of the eval runs of one controller/interface.

Scans the SAC-ZOP checkpoint directories (`real_sac_zop/checkpoints/*`) and the
MPC eval directories (`real_acados/eval_logs/*`) and writes ONE stacked figure
per controller/interface: every run of every seed as a thin trace, plus their
mean across runs as a bold line with a +-1 sigma band.

One figure per controller, all seeds pooled
--------------------------------------------
The group boundary is the first-level directory under each root - e.g.
`real_sac_zop_full/`, `real_sac_zop_fullcart/`, `mpc_full_10ms_dt10/`. Any seed
subdirectory beneath that (`real_sac_zop_full/s1/` ... `/s6/`, `real_sac/seed4/`)
is searched recursively and its runs are pooled into the SAME figure - a run
from s1 and a run from s6 both count as one more "individual run" in the
`real_sac_zop_full` plot, they are not split out by seed. Two directories that
are genuinely different sessions (e.g. the 10 ms vs. 50 ms MPC eval, which are
separate top-level directories, not seeds of one directory) still get separate
figures, since that distinction is a different control rate, not a different
seed of the same run.

    python -m bachelor.acados_cartpole.plot_eval_runs                  # everything
    python -m bachelor.acados_cartpole.plot_eval_runs --out figures --format pdf
    python -m bachelor.acados_cartpole.plot_eval_runs --dirs path/to/one_dir

Which runs are plotted
----------------------
Only runs that completed the full episode: anything that tripped or ended before
`--min-duration` seconds is dropped, with the reason printed. Failed runs are not
averageable - a run that ends at 1.5 s has no data over 85% of the window, and
its cumulative curves would freeze at whatever they had reached. Pass
`--keep-incomplete` to plot them anyway.

Schema differences
------------------
The writers do not all log the same columns (vanilla SAC has no solver stats,
SAC-ZOP logs `param`/`actor_called`, MPC_REAL logs `theta_unwrapped`/
`host_solve_ms`). Only the columns a directory actually has are plotted; panels
whose source column is missing from any run of that directory are skipped with a
note, so a new writer with a different header still produces a figure. `x_m` is
derived from the encoder counts and `cum_reward` from `reward` where absent.

Averaging across runs
---------------------
Runs are resampled onto a common time grid (spacing = median control period of
the directory) before averaging, since the sample times differ between runs.

The wrapped angle is averaged ON THE CIRCLE: the arithmetic mean of theta across
the +-pi seam is meaningless, and a hanging pole (half the runs near +pi, half
near -pi) would average to 0 rad, i.e. "perfectly upright". `|theta|` and its
running sum have no such subtlety and are the ones to read for "how far from
upright was the pole in total".

Cumulative panels (running sums) are per-sample, so a 10 ms session would show
~5x the total of a 50 ms one for identical behaviour. They are therefore scaled
by `dt / --cum-ref-dt`, i.e. expressed in units of one 50 ms sample: the 10 ms
sets get divided by ~5.6 (their measured period is ~8.9 ms), the 50 ms sets stay
as they are, and the numbers become directly comparable. `--cum-ref-dt 0` turns
the scaling off and plots the raw sums.
"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bachelor.acados_cartpole.my_helpers import X_TERM_M, counts_to_meters

HERE = Path(__file__).resolve().parent

# Where eval directories are looked for by default. Searched recursively, so the
# per-seed subdirectories (checkpoints/real_sac_zop_full/s4, real_sac/seed4, ...)
# each become their own figure.
DEFAULT_ROOTS = [
    HERE / "real_sac_zop" / "checkpoints",
    HERE / "real_acados" / "eval_logs",
]

UPRIGHT_THRESHOLD = 0.15  # rad, the band real_sac_zop/MPC_REAL call "upright"

# Sample period the running sums are expressed in. 50 ms is the RL decision
# period of the 50 ms sessions, so their curves are left essentially unchanged.
DEFAULT_CUM_REF_DT = 0.05


@dataclass(frozen=True)
class Panel:
    """One row of the stacked figure.

    Attributes:
        key: Column name in the per-run frame built by `build_series`.
        label: y-axis label.
        color: Colour of the mean line (individual traces are colour-coded by run).
        cumulative: Whether the quantity accumulates. Decides how a finished run
            is extrapolated when averaging (hold last value vs. NaN) and whether
            the rate normalization applies.
        circular: Whether the quantity is a wrapped angle, i.e. must be averaged
            on the circle rather than on the line.
    """

    key: str
    label: str
    color: str
    cumulative: bool = False
    circular: bool = False


PANELS: tuple[Panel, ...] = (
    Panel("x_m", "x (m)", "tab:blue"),
    Panel("theta_rad", "theta (rad)", "tab:red", circular=True),
    Panel("theta_abs", "|theta| (rad)", "tab:orange"),
    Panel("theta_abs_cumsum", "sum |theta| (rad)", "tab:brown", cumulative=True),
    Panel("u_force_N", "u_force (N)", "tab:purple"),
    Panel("reward", "reward", "tab:green"),
    Panel("cum_reward", "cumulative reward", "tab:olive", cumulative=True),
    Panel("u_force_cumsum", "sum |u_force| (N)", "tab:pink", cumulative=True),
    Panel("u_pwm_cumsum", "sum |u_pwm|", "dimgray", cumulative=True),
)


# READING ######################################################################
def read_trailer(path: Path) -> dict[str, str]:
    """Parse the '# key: value' summary lines appended after the last CSV row."""
    trailer: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") and ":" in line:
                key, _, value = line[1:].partition(":")
                trailer[key.strip()] = value.strip()
    return trailer


def run_dt(df: pd.DataFrame) -> float:
    """Median control period of one run, measured from its own timestamps."""
    t = df["time_s"].to_numpy(dtype=float)
    return float(np.median(np.diff(t))) if t.size > 1 else float("nan")


def build_series(df: pd.DataFrame, cum_scale: float = 1.0) -> pd.DataFrame:
    """Derive the plotted quantities from one run's raw trajectory frame.

    Every quantity whose source column is missing is simply left out - the
    directory's panel list is intersected over its runs afterwards. `x_m` is
    reconstructed from the encoder counts and `cum_reward` from the per-step
    reward where a writer did not log them.

    Args:
        df: One run as read from its CSV.
        cum_scale: Factor applied to every cumulative quantity, converting the
            per-sample sums to a common sample period (see module docstring).
    """
    out = pd.DataFrame()
    out["time_s"] = df["time_s"].astype(float)

    if "x_m" in df:
        out["x_m"] = df["x_m"].astype(float)
    elif "x_counts" in df:
        out["x_m"] = df["x_counts"].astype(float).map(counts_to_meters)

    if "theta_rad" in df:
        theta = df["theta_rad"].astype(float)
        out["theta_rad"] = theta
        # distance from upright, and its running total: the "how much angle did
        # the pole accumulate away from upright" curve
        out["theta_abs"] = theta.abs()
        out["theta_abs_cumsum"] = out["theta_abs"].cumsum() * cum_scale

    if "u_force_N" in df:
        out["u_force_N"] = df["u_force_N"].astype(float)
        # abs, not signed: a controller pushing back and forth sums to ~0 either
        # way, which hides how hard it actually worked - this is total effort
        out["u_force_cumsum"] = out["u_force_N"].abs().cumsum() * cum_scale

    if "u_pwm" in df:
        out["u_pwm_cumsum"] = df["u_pwm"].astype(float).abs().cumsum() * cum_scale

    if "reward" in df:
        out["reward"] = df["reward"].astype(float)
    if "cum_reward" in df:
        out["cum_reward"] = df["cum_reward"].astype(float) * cum_scale
    elif "reward" in df:
        out["cum_reward"] = out["reward"].cumsum() * cum_scale

    return out


def run_index(path: Path) -> int:
    """Sort key from the filename, so run10 does not land between run1 and run2."""
    match = re.search(r"run(\d+)", path.name)
    return int(match.group(1)) if match else 0


def is_complete(
    df: pd.DataFrame, trailer: dict[str, str], min_duration: float
) -> tuple[bool, str]:
    """Whether a run finished its episode, plus the reason when it did not."""
    duration = float(df["time_s"].iloc[-1])
    try:
        tripped = bool(int(float(trailer.get("terminated_by_trip", "0"))))
    except ValueError:
        tripped = False
    if not tripped and "tripped" in df:
        tripped = bool(df["tripped"].astype(float).max() > 0)

    if tripped:
        return False, f"tripped at {duration:.2f} s"
    if duration < min_duration:
        return False, f"ended after {duration:.2f} s"
    return True, ""


def load_group(
    directory: Path, min_duration: float, cum_ref_dt: float, keep_incomplete: bool
) -> tuple[list[pd.DataFrame], list[dict], list[Path], list[str]]:
    """Read the complete eval runs of one controller/interface, all seeds pooled.

    Searches `directory` recursively, so every seed subdirectory underneath it
    (s0, s1, seed4, ...) contributes its runs to the same figure - the grouping
    unit is the controller/interface, not the individual seed.

    Returns:
        (series, trailers, paths, dropped) - the kept runs and a human-readable
        line per rejected run.
    """
    series, trailers, paths, dropped = [], [], [], []
    files = sorted(
        (p for p in directory.rglob("eval_*.csv") if p.name != "eval_summary.csv"),
        key=lambda p: (run_index(p), str(p)),
    )
    for path in files:
        label = str(path.relative_to(directory))  # e.g. "s4/eval_full_run2_seed4.csv"
        try:
            df = pd.read_csv(path, comment="#")
        except Exception as e:
            dropped.append(f"{label}: unreadable ({e})")
            continue
        if df.empty or "time_s" not in df:
            dropped.append(f"{label}: no trajectory rows")
            continue

        complete, reason = is_complete(df, read_trailer(path), min_duration)
        if not complete and not keep_incomplete:
            dropped.append(f"{label}: {reason}")
            continue

        dt = run_dt(df)
        scale = (dt / cum_ref_dt) if (cum_ref_dt > 0 and np.isfinite(dt)) else 1.0
        series.append(build_series(df, cum_scale=scale))
        trailers.append(read_trailer(path))
        paths.append(path)
    return series, trailers, paths, dropped


def available_panels(series: list[pd.DataFrame]) -> tuple[Panel, ...]:
    """The panels every run of the group has the data for."""
    shared = set(series[0].columns)
    for s in series[1:]:
        shared &= set(s.columns)
    return tuple(p for p in PANELS if p.key in shared)


# AVERAGING ####################################################################
def common_grid(series: list[pd.DataFrame]) -> np.ndarray:
    """Uniform time grid spanning the longest run, spaced like the control period."""
    steps = np.concatenate([np.diff(s["time_s"].to_numpy()) for s in series if len(s) > 1])
    dt = float(np.median(steps)) if steps.size else 0.01
    end = max(float(s["time_s"].iloc[-1]) for s in series)
    return np.arange(0.0, end + dt, dt)


def resample(t: np.ndarray, y: np.ndarray, grid: np.ndarray, cumulative: bool) -> np.ndarray:
    """Put one run's series on the common grid.

    `np.interp` clamps outside the run's own time span, which is what a cumulative
    quantity needs on the right (the finished run keeps its total) and what every
    quantity needs on the tiny gap before the first sample. For an instantaneous
    quantity the tail past the run's end is set to NaN instead: the run is over,
    it should not pull the mean towards its last value.
    """
    out = np.interp(grid, t, y)
    if not cumulative:
        out[grid > t[-1]] = np.nan
    return out


def mean_across_runs(
    stack: np.ndarray, min_runs: int, circular: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, std and contributing-run count per grid point.

    A wrapped angle needs `circular=True`: the arithmetic mean of theta is
    meaningless across the +-pi seam (see module docstring). The circular mean is
    the angle of the summed unit vectors, and the spread is the circular standard
    deviation sqrt(-2 ln R), which diverges as the runs spread evenly around the
    circle - clipped to pi here so the band stays on-axis.

    Blanks out grid points backed by fewer than `min_runs` runs, so the average
    stops where it would otherwise degenerate into a single trace.
    """
    n_valid = np.sum(~np.isnan(stack), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        if circular:
            cos_mean = np.nanmean(np.cos(stack), axis=0)
            sin_mean = np.nanmean(np.sin(stack), axis=0)
            mean = np.arctan2(sin_mean, cos_mean)
            resultant = np.clip(np.hypot(cos_mean, sin_mean), 1e-12, 1.0)
            std = np.minimum(np.sqrt(-2.0 * np.log(resultant)), np.pi)
        else:
            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
    blank = n_valid < min_runs
    mean[blank] = np.nan
    std[blank] = np.nan
    return mean, std, n_valid


# PLOTTING #####################################################################
def plot_group(
    name: str,
    series: list[pd.DataFrame],
    trailers: list[dict],
    out_dir: Path,
    n_dropped: int = 0,
    cum_ref_dt: float = DEFAULT_CUM_REF_DT,
    formats: tuple[str, ...] = ("png",),
    min_runs: int = 2,
    show: bool = False,
) -> list[Path]:
    """Write one stacked figure with every run of a variant plus their mean."""
    n_runs = len(series)
    panels = available_panels(series)
    if not panels:
        print("  no plottable columns in common - skipped")
        return []
    missing = [p.label for p in PANELS if p not in panels]
    if missing:
        print(f"  panels without data here: {', '.join(missing)}")

    grid = common_grid(series)
    min_runs = max(1, min(min_runs, n_runs))
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, n_runs))

    fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(11, 1.75 * len(panels)))
    axes = np.atleast_1d(axes)

    for ax, panel in zip(axes, panels):
        stack = np.vstack([
            resample(
                s["time_s"].to_numpy(), s[panel.key].to_numpy(), grid, panel.cumulative
            )
            for s in series
        ])

        for row, color in zip(stack, colors):
            ax.plot(grid, row, color=color, linewidth=0.7, alpha=0.45)

        mean, std, _ = mean_across_runs(stack, min_runs, circular=panel.circular)
        ax.fill_between(grid, mean - std, mean + std, color=panel.color, alpha=0.18,
                        linewidth=0)
        ax.plot(grid, mean, color=panel.color, linewidth=2.0)

        ax.set_ylabel(panel.label, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)

        if panel.key == "x_m":
            for sign in (-1.0, 1.0):
                ax.axhline(sign * float(X_TERM_M), color="r", linestyle="--",
                           alpha=0.5, linewidth=0.8)
        if panel.key in ("theta_rad", "theta_abs"):
            ax.axhspan(-UPRIGHT_THRESHOLD, UPRIGHT_THRESHOLD, color="g", alpha=0.12)
            ax.axhline(0.0, color="k", linestyle="--", alpha=0.3, linewidth=0.8)

    axes[-1].set_xlabel("time (s)", fontsize=10)

    # one legend for the whole figure, above the panels: the traces mean the same
    # thing in every row, and inside an axes it would sit on top of the data
    handles = [
        plt.Line2D([], [], color=colors[len(colors) // 2], linewidth=0.9, alpha=0.6,
                   label=f"individual runs (n={n_runs})"),
        plt.Line2D([], [], color="k", linewidth=2.0,
                   label=f"mean (>= {min_runs} run{'s' if min_runs > 1 else ''})"),
        plt.Line2D([], [], color="k", linewidth=6, alpha=0.18, label="+-1 sigma"),
    ]

    fig.suptitle(
        f"{name} - {n_runs} complete eval runs\n"
        f"{summary_line(trailers, cum_scale(series, cum_ref_dt), cum_ref_dt)}\n"
        f"{method_line(series, n_dropped, cum_ref_dt)}",
        fontsize=12, fontweight="bold", y=0.998,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.legend(handles=handles, fontsize=9, loc="upper center",
               bbox_to_anchor=(0.5, 0.952), ncol=3, framealpha=0.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        target = out_dir / f"{name}_eval_overview.{fmt}"
        fig.savefig(target, bbox_inches="tight", dpi=150)
        written.append(target)
        print(f"  wrote {target}")

    if show:
        plt.show()
    plt.close(fig)
    return written


def _floats(trailers: list[dict], key: str) -> np.ndarray:
    """Every parseable float under `key`, across the trailers of one group."""
    values = []
    for trailer in trailers:
        try:
            values.append(float(trailer[key]))
        except (KeyError, ValueError, TypeError):
            continue
    return np.asarray(values, dtype=float)


def summary_line(trailers: list[dict], cum_scale: float = 1.0,
                 cum_ref_dt: float = DEFAULT_CUM_REF_DT) -> str:
    """Context line under the title: return and how many runs stabilized.

    `episodic_return` in the trailers is a per-env-step sum, so it carries the
    same rate dependence as the cumulative panels and is scaled the same way -
    otherwise the number in the title would contradict the curve below it.
    """
    parts = []
    returns = _floats(trailers, "episodic_return") * cum_scale
    if returns.size:
        unit = f" (per {cum_ref_dt * 1000:.0f} ms)" if cum_scale != 1.0 else ""
        parts.append(f"return{unit} {returns.mean():.1f} +- {returns.std():.1f}")
    stabilized = _floats(trailers, "stabilized")
    if stabilized.size:
        parts.append(f"stabilized {int(stabilized.sum())}/{stabilized.size}")
    upright = _floats(trailers, "upright_fraction")
    if upright.size:
        parts.append(f"upright fraction {upright.mean():.2f}")
    return "   ".join(parts)


def cum_scale(series: list[pd.DataFrame], cum_ref_dt: float) -> float:
    """Factor that puts a group's per-sample sums on the reference sample period."""
    if cum_ref_dt <= 0:
        return 1.0
    dt = float(np.median([run_dt(s) for s in series]))
    return dt / cum_ref_dt if np.isfinite(dt) else 1.0


def method_line(series: list[pd.DataFrame], n_dropped: int, cum_ref_dt: float) -> str:
    """Second context line: what was dropped and how the panels were scaled."""
    dt = float(np.median([run_dt(s) for s in series]))
    parts = [f"dt = {dt * 1000:.1f} ms"]
    if n_dropped:
        parts.append(f"{n_dropped} incomplete run(s) dropped")
    if cum_ref_dt > 0:
        parts.append(f"cumulative panels x {dt / cum_ref_dt:.2f} (per {cum_ref_dt * 1000:.0f} ms)")
    parts.append("theta: circular mean")
    return "   ".join(parts)


def print_table(rows: list[tuple[str, list[dict], int, float, float]]) -> None:
    """Compact per-variant summary of the trailer metrics, for the terminal.

    The return is rate-normalized like the cumulative panels, so the 10 ms and
    50 ms variants can be read off the same column; `upright` (a fraction) and
    `stab` (a count) are rate-independent already.
    """
    if not rows:
        return
    header = (f"{'variant':<40}{'runs':>5}{'drop':>6}{'dt/ms':>7}"
              f"{'return/50ms':>19}{'upright':>9}{'stab':>7}")
    print("\n" + header)
    print("-" * len(header))
    for name, trailers, n_dropped, dt, scale in rows:
        returns = _floats(trailers, "episodic_return") * scale
        upright = _floats(trailers, "upright_fraction")
        stabilized = _floats(trailers, "stabilized")
        ret = f"{returns.mean():9.2f}+-{returns.std():<7.2f}" if returns.size else " " * 18
        upr = f"{upright.mean():9.2f}" if upright.size else " " * 9
        stb = f"{int(stabilized.sum())}/{stabilized.size}" if stabilized.size else "-"
        print(f"{name:<40}{len(trailers):>5}{n_dropped:>6}{dt * 1000:>7.1f}"
              f"{ret:>19}{upr}{stb:>7}")


# ENTRY POINT ##################################################################
def group_name(directory: Path) -> str:
    """Figure name for a group directory - just its own name.

    A group directory is always a first-level subdirectory of one of the roots
    (see `discover`), so its name already identifies the controller/interface;
    the seed subdirectories merged into it are not part of the name.
    """
    return directory.name


def discover(roots: list[Path]) -> list[Path]:
    """One group per controller/interface: the first-level subdirectory of each
    root that holds eval CSVs, directly or in any seed subdirectory beneath it.

    E.g. `real_sac_zop_full/` (5 runs directly) and `real_sac_zop_full/s1/` ...
    `/s6/` (more runs, one seed each) all fall under the same group directory
    `real_sac_zop_full/`, so `load_group`'s recursive search pools every seed of
    that controller into the one figure - this is what makes "all seeds in one
    plot" work without the caller needing to enumerate seed subdirectories.
    """
    found: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            print(f"(no such directory: {root})")
            continue
        for path in root.rglob("eval_*.csv"):
            if path.name == "eval_summary.csv":
                continue
            group_dir = root / path.relative_to(root).parts[0]
            found.add(group_dir)
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--dirs", nargs="+", type=Path,
        help="Eval directories to plot. Default: every directory under "
             "real_sac_zop/checkpoints and real_acados/eval_logs with eval CSVs.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Collect all figures here. Default: next to each directory's CSVs.",
    )
    parser.add_argument(
        "--format", nargs="+", default=["png"], choices=["png", "pdf", "svg"],
        help="Output formats (default: png).",
    )
    parser.add_argument(
        "--min-duration", type=float, default=9.5,
        help="Drop runs shorter than this many seconds (default: 9.5, i.e. keep "
             "the 10 s episodes and drop everything that ended early).",
    )
    parser.add_argument(
        "--keep-incomplete", action="store_true",
        help="Also plot tripped/truncated runs instead of dropping them.",
    )
    parser.add_argument(
        "--cum-ref-dt", type=float, default=DEFAULT_CUM_REF_DT,
        help="Sample period the cumulative panels are expressed in, in seconds "
             "(default: 0.05). 0 disables the scaling and plots raw sums.",
    )
    parser.add_argument(
        "--min-runs", type=int, default=2,
        help="Draw the mean only where at least this many runs are still "
             "running (default: 2).",
    )
    parser.add_argument("--show", action="store_true", help="Show each figure.")
    args = parser.parse_args()

    directories = args.dirs if args.dirs else discover(DEFAULT_ROOTS)
    if not directories:
        raise SystemExit("No eval directories with eval_*.csv found.")

    summary: list[tuple[str, list[dict], int, float, float]] = []
    for directory in directories:
        name = group_name(directory)
        print(f"\n{name}  ({directory})")
        series, trailers, paths, dropped = load_group(
            directory, args.min_duration, args.cum_ref_dt, args.keep_incomplete
        )
        for line in dropped:
            print(f"  dropped {line}")
        if not series:
            print("  no complete eval runs - skipped")
            continue
        run_labels = ", ".join(str(p.relative_to(directory)) for p in paths)
        print(f"  {len(series)} runs: {run_labels}")
        plot_group(
            name=name,
            series=series,
            trailers=trailers,
            out_dir=args.out if args.out else directory,
            n_dropped=len(dropped),
            cum_ref_dt=args.cum_ref_dt,
            formats=tuple(args.format),
            min_runs=args.min_runs,
            show=args.show,
        )
        summary.append((
            name, trailers, len(dropped),
            float(np.median([run_dt(s) for s in series])),
            cum_scale(series, args.cum_ref_dt),
        ))

    print_table(summary)


if __name__ == "__main__":
    main()
