"""Policy over the state space: what the SAC-ZOP controllers and plain SAC learned.

For the evaluated checkpoint of every controller, the deterministic policy is evaluated on a grid of
cart position p and pole angle theta (theta = 0 upright), in a 3x3 array of
panels for the cart velocity pdot (columns) and the pole angular velocity
thetadot (rows). Four figures per SAC-ZOP controller go to paper_plots/ as
`policy_<tag>_<what>.pdf`:

    param    learned MPC parameter phi_psi(s): theta_ref for Angle and
             Angle-10, p_ref for Pos and Cart, on a colour scale
             spanning its bounds
    force    applied force F = u0*(x, phi_psi(s)) of the hierarchical policy
    nominal  force of the same OCP with phi = 0, i.e. the prior on its own
    critic   Q_w(s, phi_psi(s)), the minimum over the two critics

Plain SAC has no MPC, so it gets two: `force`, the actor's force F = pi(s), and
`critic`, Q_w(s, pi(s)).

The grids are cached in paper_plots/policy_cache/<tag>.npz. From them, three stacked
figures show one map per controller, two controllers per row:
`policy_stacked_{critic,force,param}.pdf` (param without SAC, force led by the nominal
MPC: the full OCP with phi = 0, the same at 10 and 50 ms). `--stacked-only` draws
them from the cache without evaluating anything:

    python -m bachelor.acados_cartpole.plot_policy_statespace --stacked-only

Names and symbols follow main_paper.tex: s = (p, theta, pdot, thetadot), force
F in [-20, 20] N, controller names from `plot_training_curves.PAPER_NAMES`. The two
force maps share the colour scale of the input bounds, the parameter maps that
of the parameter bounds, so all controllers read on the same scale.

Each OCP is solved from a cold start at the grid state, so the force maps show
what one capped SQP solve returns there, not the warm-started hardware loop. The
share of converged solves is printed per controller, for the caption.

The figures are drawn at the text width of the paper in its fonts (`paper_font`).

The checkpoint is, per controller, the one whose eval runs the trajectory figure and
the eval table show (`plot_benchmark.TABLE_CONTROLLERS`): the seed directory that holds
those runs. --seed overrides it.

acados has to be importable: the venv still points at the old repo path, so
export ACADOS_SOURCE_DIR, ACADOS_INSTALL_DIR, LD_LIBRARY_PATH and PYTHONPATH as in
simulation_zaczop/run_bufferfill10k.sh. The solvers are generated into a
temporary directory.

    python -m bachelor.acados_cartpole.plot_policy_statespace
    python -m bachelor.acados_cartpole.plot_policy_statespace --controllers zop_cart sac
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MultipleLocator

from bachelor.acados_cartpole.my_helpers import X_TERM_M
from bachelor.acados_cartpole.my_planner_registry import make_planner
from bachelor.acados_cartpole.plot_benchmark import TABLE_CONTROLLERS, pi_formatter
from bachelor.acados_cartpole.plot_training_curves import NAMES, PAPER_NAMES, TEXTWIDTH_IN, paper_font
from bachelor.acados_cartpole.real_sac_zop.my_sac import SacActor, SacCritic, SacTrainerConfig
from bachelor.acados_cartpole.real_sac_zop.my_sac_zop import MpcSacActor, SacZopTrainerConfig
from leap_c.planner import ControllerFromPlanner
from leap_c.torch.nn.extractor import get_extractor_cls

HERE = Path(__file__).resolve().parent
CHECKPOINTS = HERE / "real_sac_zop" / "checkpoints"

XDOTS = (-1.0, 0.0, 1.0)  # m/s, panel columns
THETADOTS = (5.0, 0.0, -5.0)  # rad/s, panel rows from top to bottom
X_RANGE = (-0.37, 0.37)  # m, the state constraint set of the OCPs
SOLVE_CHUNK = 256  # N_batch_max of the acados batch solver


@dataclass(frozen=True)
class Variant:
    directory: str  # below CHECKPOINTS, also the key into NAMES
    tag: str  # file name part
    planner: str | None  # my_planner_registry name, None for plain SAC
    param_symbol: str = ""  # LaTeX, as in Table "Compared controllers" of the paper
    param_unit: str = ""
    seed_prefix: str = "s"  # seed directories are <prefix><n>

    @property
    def label(self) -> str:
        name = NAMES[self.directory]
        return PAPER_NAMES.get(name, name)


VARIANTS = (
    Variant("real_sac_zop_full", "zop_angle", "full", r"\theta_{\mathrm{ref}}", "rad"),
    Variant("real_sac_zop_full_10ms", "zop_angle10", "full", r"\theta_{\mathrm{ref}}", "rad"),
    Variant("real_sac_zop_fullcart", "zop_pos", "fullcart", r"p_{\mathrm{ref}}", "m"),
    Variant("real_sac_zop_cart", "zop_cart", "cart", r"p_{\mathrm{ref}}", "m"),
    Variant("real_sac", "sac", None, seed_prefix="seed"),
)


def eval_checkpoint(variant: Variant) -> Path:
    """The seed directory of the controller's eval runs in `plot_benchmark`: its eval
    directory if that holds the networks, else the one directory below it that holds
    both the eval runs and the networks."""
    controller = next(c for c in TABLE_CONTROLLERS if c.label == NAMES[variant.directory])
    if (controller.directory / "actor.pth").exists():
        return controller.directory
    dirs = {p.parent for p in controller.directory.rglob("eval_*.csv")
            if (p.parent / "actor.pth").exists()}
    if len(dirs) != 1:
        raise RuntimeError(f"{variant.label}: no unique checkpoint below {controller.directory}: "
                           f"{sorted(map(str, dirs))}")
    return dirs.pop()


OBS_SPACE = gym.spaces.Box(  # as in real_sac_zop.py and real_sac.py
    low=np.array([-X_TERM_M, -np.pi, -5, -21], dtype=np.float32),
    high=np.array([X_TERM_M, np.pi, 5, 21], dtype=np.float32),
    dtype=np.float32,
)
SAC_FORCE_MAX = 20.0  # N, the action bound of real_sac.py


def load_networks(seed_dir: Path, controller: ControllerFromPlanner):
    """Actor and critic of one seed, built as in real_sac_zop.py."""
    obs_space = OBS_SPACE
    cfg = SacZopTrainerConfig()
    cfg.critic_mlp.norm_layer = "layer_norm"
    extractor_cls = get_extractor_cls("identity")
    actor = MpcSacActor(
        extractor_cls=extractor_cls,
        observation_space=obs_space,
        controller=controller,
        distribution_name=cfg.distribution_name,
        mlp_cfg=cfg.actor_mlp,
        init_param_with_default=cfg.init_param_with_default,
    )
    critic = SacCritic(
        extractor_cls=extractor_cls,
        observation_space=obs_space,
        action_space=controller.param_space,
        mlp_cfg=cfg.critic_mlp,
        num_critics=cfg.num_critics,
    )
    actor.load_state_dict(torch.load(seed_dir / "actor.pth", map_location="cpu"))
    critic.load_state_dict(torch.load(seed_dir / "critic.pth", map_location="cpu"))
    return actor.eval(), critic.eval()


def load_sac_networks(seed_dir: Path):
    """Actor and critic of one plain SAC seed, built as in real_sac.py."""
    action_space = gym.spaces.Box(low=np.array([-SAC_FORCE_MAX], dtype=np.float32),
                                  high=np.array([SAC_FORCE_MAX], dtype=np.float32),
                                  dtype=np.float32)
    cfg = SacTrainerConfig()
    cfg.critic_mlp.norm_layer = "layer_norm"
    extractor_cls = get_extractor_cls("identity")
    actor = SacActor(extractor_cls=extractor_cls, observation_space=OBS_SPACE,
                     action_space=action_space, distribution_name=cfg.distribution_name,
                     mlp_cfg=cfg.actor_mlp)
    critic = SacCritic(extractor_cls=extractor_cls, observation_space=OBS_SPACE,
                       action_space=action_space, mlp_cfg=cfg.critic_mlp,
                       num_critics=cfg.num_critics)
    actor.load_state_dict(torch.load(seed_dir / "actor.pth", map_location="cpu"))
    critic.load_state_dict(torch.load(seed_dir / "critic.pth", map_location="cpu"))
    return actor.eval(), critic.eval()


def mpc_force(planner, obs: torch.Tensor, param: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """First OCP input u0 for every state, cold-started, in chunks the batch solver
    takes, and whether each solve converged (acados status 0)."""
    forces, converged = [], []
    for i in range(0, len(obs), SOLVE_CHUNK):
        ctx, u0, *_ = planner(obs[i:i + SOLVE_CHUNK], param=param[i:i + SOLVE_CHUNK], ctx=None)
        forces.append(u0.detach().cpu().numpy().ravel())
        converged.append(np.asarray(ctx.status).ravel() == 0)
    return np.concatenate(forces), np.concatenate(converged)


def grid_obs(x_vals: np.ndarray, theta_vals: np.ndarray):
    """(row, column, observations) per panel, and the shape of one map."""
    X, TH = np.meshgrid(x_vals, theta_vals)
    panels = []
    for r, thetadot in enumerate(THETADOTS):
        for c, xdot in enumerate(XDOTS):
            obs = torch.tensor(
                np.stack([X.ravel(), TH.ravel(), np.full(X.size, xdot), np.full(X.size, thetadot)],
                         axis=1),
                dtype=torch.float32,
            )
            panels.append((r, c, obs))
    return panels, X.shape


def evaluate_sac(actor, critic, x_vals: np.ndarray, theta_vals: np.ndarray):
    """Force and Q of a plain SAC policy on the grid, shaped as in `evaluate`."""
    panels, shape = grid_obs(x_vals, theta_vals)
    out = {key: np.empty((len(THETADOTS), len(XDOTS), *shape)) for key in ("force", "critic")}
    for r, c, obs in panels:
        with torch.no_grad():
            action, _, _ = actor(obs, deterministic=True)
            q = critic(obs, action).min(dim=1).values
        out["force"][r, c] = action.cpu().numpy().reshape(shape)
        out["critic"][r, c] = q.cpu().numpy().reshape(shape)
    return out


def evaluate(actor, critic, planner, x_vals: np.ndarray, theta_vals: np.ndarray):
    """Parameter, policy force, nominal force and Q on the grid, each of shape
    (len(THETADOTS), len(XDOTS), len(theta_vals), len(x_vals)), plus the fraction
    of converged OCP solves behind each force map."""
    panels, shape = grid_obs(x_vals, theta_vals)
    out = {key: np.empty((len(THETADOTS), len(XDOTS), *shape))
           for key in ("param", "force", "nominal", "critic")}
    converged = {"force": [], "nominal": []}
    for r, c, obs in panels:
        with torch.no_grad():
            param = actor(obs, ctx=None, deterministic=True, only_param=True).param
            q = critic(obs, param).min(dim=1).values  # (batch, num_critics) -> (batch,)
        out["param"][r, c] = param.cpu().numpy().reshape(shape)
        out["critic"][r, c] = q.cpu().numpy().reshape(shape)
        for key, p in (("force", param), ("nominal", torch.zeros_like(param))):
            force, ok = mpc_force(planner, obs, p)
            out[key][r, c] = force.reshape(shape)
            converged[key].append(ok)
    return out, {key: float(np.mean(np.concatenate(oks))) for key, oks in converged.items()}


def panel_figure(data: np.ndarray, x_vals, theta_vals, title: str, cbar_label: str,
                 cmap: str, norm: Normalize) -> plt.Figure:
    """3 x 3 maps at the text width of the paper, saved at that size, so the fonts are
    those of the other paper figures (`paper_font`)."""
    fig, axes = plt.subplots(3, 3, figsize=(TEXTWIDTH_IN, 4.4), sharex=True, sharey=True,
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.03)
    extent = (x_vals[0], x_vals[-1], theta_vals[0], theta_vals[-1])
    for r, thetadot in enumerate(THETADOTS):
        for c, xdot in enumerate(XDOTS):
            ax = axes[r, c]
            im = ax.imshow(data[r, c], origin="lower", extent=extent, aspect="auto",
                           cmap=cmap, norm=norm, interpolation="nearest", rasterized=True)
            if r == 0:
                ax.set_title(rf"$\dot p = {xdot:g}$ m/s")
            if c == 0:
                ax.set_ylabel(rf"$\dot\theta = {thetadot:g}$ rad/s" + "\n" + r"$\theta$ [rad]")
            if r == len(THETADOTS) - 1:
                ax.set_xlabel(r"$p$ [m]")
    axes[0, 0].yaxis.set_major_locator(MultipleLocator(np.pi))
    axes[0, 0].yaxis.set_major_formatter(FuncFormatter(pi_formatter))
    axes[0, 0].set_xticks([-0.3, 0.0, 0.3])
    fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02, aspect=30, label=cbar_label)
    fig.suptitle(title)
    return fig


CACHE = HERE / "paper_plots" / "policy_cache"
STACK_COLS = 2  # controllers per row in the stacked figures


def stacked_figure(blocks: list[dict], colorbars: list[tuple[list[int], str]], x_vals,
                   theta_vals) -> plt.Figure:
    """The 3 x 3 maps of several controllers, STACK_COLS controllers per row, at the text
    width of the paper. `blocks` hold label, data, cmap and norm per controller;
    `colorbars` the block indices per colour bar and its label, each bar right of the
    rightmost of its blocks and over their rows, with the cmap and norm of its first
    block. Only the outer panels carry tick labels; the panel velocities are noted once,
    in grey."""
    n = len(blocks)
    rows = -(-n // STACK_COLS)
    # layout in inches; room for a colour bar after every column that has one
    width, left, bottom, cbar_room = TEXTWIDTH_IN, 0.62, 0.36, 0.62
    header, title, block_gap, row_gap, gap, vgap = 0.17, 0.18, 0.14, 0.10, 0.05, 0.09
    panel_h = 0.56
    bar_cols = {max(i % STACK_COLS for i in indices) for indices, _ in colorbars}
    room = [cbar_room if col in bar_cols else 0.0 for col in range(STACK_COLS)]
    block_w = (width - left - sum(room) - (STACK_COLS - 1) * block_gap) / STACK_COLS
    panel_w = (block_w - 2 * gap) / 3
    block_h = 3 * panel_h + 2 * vgap
    col_x = [left + col * (block_w + block_gap) + sum(room[:col]) for col in range(STACK_COLS)]
    height = header + rows * (title + block_h) + (rows - 1) * row_gap + bottom
    fig = plt.figure(figsize=(width, height))
    small = dict(fontsize=plt.rcParams["axes.titlesize"] - 2, style="italic", color="0.4")
    extent = (x_vals[0], x_vals[-1], theta_vals[0], theta_vals[-1])

    def box(x0, y0, w, h):  # inches from bottom left -> figure fraction
        return (x0 / width, y0 / height, w / width, h / height)

    tops = {}
    for i, block in enumerate(blocks):
        row, col = divmod(i, STACK_COLS)
        x0 = col_x[col]
        top = height - header - title - row * (title + block_h + row_gap)
        tops[i] = (top - block_h, top)
        below = i + STACK_COLS < n  # another block under this one
        for r, thetadot in enumerate(THETADOTS):
            for c, xdot in enumerate(XDOTS):
                ax = fig.add_axes(box(x0 + c * (panel_w + gap), top - (r + 1) * panel_h - r * vgap,
                                      panel_w, panel_h))
                ax.imshow(block["data"][r, c], origin="lower", extent=extent, aspect="auto",
                          cmap=block["cmap"], norm=block["norm"], interpolation="nearest",
                          rasterized=True)
                ax.yaxis.set_major_locator(MultipleLocator(np.pi))
                ax.yaxis.set_major_formatter(FuncFormatter(pi_formatter))
                ax.set_xticks([-0.2, 0.0, 0.2])
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"${v:g}$"))
                ax.tick_params(labelleft=(col == 0 and c == 0),
                               labelbottom=(r == 2 and not below))
                if r == 0 and c == 0:
                    ax.set_title(block["label"], loc="left")
                if row == 0 and r == 0:
                    fig.text((x0 + c * (panel_w + gap) + panel_w / 2) / width,
                             1 - 0.02 / height, rf"$\dot p = {xdot:g}$ m/s", ha="center",
                             va="top", **small)
                if col == 0 and c == 0:
                    fig.text(0.24 / width, (top - (r + 0.5) * panel_h - r * vgap) / height,
                             rf"$\dot\theta = {thetadot:g}$ rad/s", rotation=90, ha="center",
                             va="center", **small)
                if r == 2 and c == 1 and not below:
                    ax.set_xlabel(r"$p$ [m]")
    fig.text(0.04 / width, 0.5 + (bottom - header - title) / 2 / height, r"$\theta$ [rad]",
             rotation=90, ha="left", va="center")

    for indices, label in colorbars:
        lo = min(tops[i][0] for i in indices)
        hi = max(tops[i][1] for i in indices)
        col = max(i % STACK_COLS for i in indices)
        cax = fig.add_axes(box(col_x[col] + block_w + 0.08, lo, 0.1, hi - lo))
        first = blocks[indices[0]]
        fig.colorbar(plt.cm.ScalarMappable(norm=first["norm"], cmap=first["cmap"]), cax=cax,
                     label=label)
    return fig


def save_cache(variant: Variant, grids: dict, x_vals, theta_vals, bound: float) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / f"{variant.tag}.npz", x_vals=x_vals, theta_vals=theta_vals, bound=bound,
             **grids)


def draw_stacked(args) -> None:
    """The three stacked figures from the cached grids of all controllers."""
    cached = {v.tag: np.load(CACHE / f"{v.tag}.npz") for v in VARIANTS}
    first = next(iter(cached.values()))
    x_vals, theta_vals = first["x_vals"], first["theta_vals"]
    force_norm = Normalize(-SAC_FORCE_MAX, SAC_FORCE_MAX)
    # Q on its own scale per controller: the ranges differ too much (SAC about 10-120,
    # Angle-10 about 250-390) for one scale to show the structure of each
    def q_norm(v):
        q = cached[v.tag]["critic"]
        return Normalize(q.min(), q.max())

    def blocks(what, norm_of, cmap, variants=VARIANTS):
        return [dict(label=v.label, data=cached[v.tag][what], cmap=cmap, norm=norm_of(v))
                for v in variants]

    zop = [v for v in VARIANTS if v.planner is not None]
    figures = {
        "critic": (blocks("critic", q_norm, "Blues"),
                   [([i], r"$Q$") for i in range(len(VARIANTS))]),
        # led by the nominal MPC: the full OCP with phi = 0, the `nominal` map of a
        # full-model variant. It is the same OCP at 10 and 50 ms, so the cold-started map
        # is the same for MPC (50 ms); named after the 10 ms baseline.
        "force": ([dict(label=PAPER_NAMES["MPC-10"], data=cached["zop_angle10"]["nominal"],
                        cmap="RdBu_r", norm=force_norm)]
                  + blocks("force", lambda v: force_norm, "RdBu_r"),
                  [(list(range(len(VARIANTS) + 1)), r"$F$ [N]")]),
        # one colour bar per row: the angle references in rad, the position ones in m
        "param": (blocks("param", lambda v: Normalize(-float(cached[v.tag]["bound"]),
                                                      float(cached[v.tag]["bound"])),
                         "RdBu_r", zop),
                  [([0, 1], r"$\theta_{\mathrm{ref}}$ [rad]"),
                   ([2, 3], r"$p_{\mathrm{ref}}$ [m]")]),
    }
    assert [v.param_unit for v in zop] == ["rad", "rad", "m", "m"], "param rows mix units"
    for what, (blks, colorbars) in figures.items():
        with paper_font():
            fig = stacked_figure(blks, colorbars, x_vals, theta_vals)
            for fmt in args.format:
                target = args.out / f"policy_stacked_{what}.{fmt}"
                fig.savefig(target, dpi=300)  # at its printed size, no tight bbox
                print(f"wrote {target}")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--controllers", nargs="+", choices=[v.tag for v in VARIANTS],
                        default=[v.tag for v in VARIANTS], help="Controllers to plot.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed to plot instead of the evaluated one (applies to all "
                             "selected controllers).")
    parser.add_argument("--resolution", type=int, default=61,
                        help="Grid points along x and along theta (default: 61).")
    parser.add_argument("--out", type=Path, default=HERE / "paper_plots",
                        help="Output directory (default: paper_plots/ next to this script).")
    parser.add_argument("--format", nargs="+", default=["pdf"],
                        choices=["png", "pdf", "svg"], help="Output formats.")
    parser.add_argument("--stacked-only", action="store_true",
                        help="Only draw the stacked figures, from the cached grids.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stacked_only:
        draw_stacked(args)
        return

    x_vals = np.linspace(*X_RANGE, args.resolution)
    theta_vals = np.linspace(-np.pi, np.pi, args.resolution)
    args.out.mkdir(parents=True, exist_ok=True)

    def save(grid, what: str, variant: Variant, title, cbar_label, cmap, norm) -> None:
        # also around the save: usetex and its preamble are read when the text is drawn
        with paper_font():
            fig = panel_figure(grid, x_vals, theta_vals, title, cbar_label, cmap, norm)
            for fmt in args.format:
                target = args.out / f"policy_{variant.tag}_{what}.{fmt}"
                fig.savefig(target, dpi=300)  # at its printed size, no tight bbox
                print(f"wrote {target}")
        plt.close(fig)

    with tempfile.TemporaryDirectory(prefix="policy_statespace_acados_") as export_root:
        for variant in (v for v in VARIANTS if v.tag in args.controllers):
            seed_dir = eval_checkpoint(variant)
            if args.seed is not None:
                seed_dir = CHECKPOINTS / variant.directory / f"{variant.seed_prefix}{args.seed}"
            print(f"{variant.label}: {seed_dir.relative_to(CHECKPOINTS)}")
            force_norm = Normalize(-SAC_FORCE_MAX, SAC_FORCE_MAX)

            if variant.planner is None:
                actor, critic = load_sac_networks(seed_dir)
                grids = evaluate_sac(actor, critic, x_vals, theta_vals)
                save_cache(variant, grids, x_vals, theta_vals, SAC_FORCE_MAX)
                q = grids["critic"]
                save(grids["force"], "force", variant, rf"{variant.label}: applied force $F$",
                     r"$F$ [N]", "RdBu_r", force_norm)
                save(q, "critic", variant, rf"{variant.label}: critic $Q$",
                     r"$Q$", "Blues", Normalize(q.min(), q.max()))
                continue

            planner, cfg = make_planner(variant.planner,
                                        export_directory=Path(export_root) / variant.tag)
            assert cfg.Fmax == SAC_FORCE_MAX, "all force maps share one colour scale"
            controller = ControllerFromPlanner(planner)
            actor, critic = load_networks(seed_dir, controller)
            grids, converged = evaluate(actor, critic, planner, x_vals, theta_vals)
            print(f"  converged OCP solves: policy {converged['force']:.0%}, "
                  f"phi = 0 {converged['nominal']:.0%}")

            sym, unit = variant.param_symbol, variant.param_unit
            bound = float(controller.param_space.high[0])
            save_cache(variant, grids, x_vals, theta_vals, bound)
            q = grids["critic"]
            figures = {
                "param": (rf"{variant.label}: learned parameter ${sym}$",
                          rf"${sym}$ [{unit}]", "RdBu_r", Normalize(-bound, bound)),
                "force": (rf"{variant.label}: applied force $F$",
                          r"$F$ [N]", "RdBu_r", force_norm),
                "nominal": (rf"{variant.label}: MPC force with ${sym} = 0$",
                            r"$F$ [N]", "RdBu_r", force_norm),
                "critic": (rf"{variant.label}: critic $Q$",
                           r"$Q$", "Blues", Normalize(q.min(), q.max())),
            }
            for what, (title, cbar_label, cmap, norm) in figures.items():
                save(grids[what], what, variant, title, cbar_label, cmap, norm)

    if all((CACHE / f"{v.tag}.npz").exists() for v in VARIANTS):
        draw_stacked(args)


if __name__ == "__main__":
    main()
