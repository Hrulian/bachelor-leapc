"""Run several sim_sac_zop.py trainings in parallel (one process per run).

Each (reward, seed) combination becomes its own process with its own
run directory, its own acados code generation and its own wandb run.
All runs of a sweep share a wandb group so they can be compared side by
side in the dashboard (group by 'reward' in the wandb UI).

Examples:
    # compare all registered rewards, 2 seeds each, on all cores
    python run_parallel.py --rewards default cosine energy --seeds 0 1

    # quick smoke test
    python run_parallel.py --rewards default --seeds 0 --steps 5000 --workers 1
"""

import os
import signal
import subprocess
import sys
import time
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

# make 'bachelor' importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from bachelor.acados_cartpole.simulation_zaczop.rewards import REWARDS
from bachelor.acados_cartpole.simulation_zaczop.planner_registry import PLANNER_REGISTRY

# ---------------------------------------------------------------------------
# Which planner (and thus OCP / parameter interface) to bind for this sweep.
# Names come from planner_registry.py:
#   "full" -> my_planner.py       (cart + pendulum, 4 states)
#   "cart" -> my_planner_cart.py  (cart only, 2 states, only cart pos learnable)
# Override per run on the command line with --planner.
PLANNER = "full"
# ---------------------------------------------------------------------------


def parse_args():
    p = ArgumentParser(description="Parallel SAC-ZOP simulation sweep")
    p.add_argument("--planner", type=str, default=PLANNER, choices=sorted(PLANNER_REGISTRY),
                   help="Which planner/OCP to bind (default set by PLANNER at top of file)")
    p.add_argument("--rewards", nargs="+", default=["default"], choices=sorted(REWARDS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                   help="Max concurrent processes (each uses ~2 threads: torch + acados)")
    p.add_argument("--torch-threads", type=int, default=1,
                   help="torch threads per process")
    p.add_argument("--wandb-project", type=str, default="cartpole-planner-ablation-sim")
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--group", type=str, default=None,
                   help="wandb group name (default: sweep_<timestamp>)")
    p.add_argument("--runs-dir", type=Path, default=Path(__file__).parent / "runs")
    # any unrecognized arguments are forwarded to sim_sac_zop.py,
    # e.g. --train-start 500 --heatmap-every 50
    args, extra = p.parse_known_args()
    args.extra = extra
    return args


def main():
    args = parse_args()
    group = args.group or f"sweep_{int(time.time())}"
    script = Path(__file__).parent / "sim_sac_zop.py"

    jobs = []
    for reward, seed in product(args.rewards, args.seeds):
        run_name = f"{group}_{args.planner}_{reward}_s{seed}"
        cmd = [
            sys.executable, str(script),
            "--planner", args.planner,
            "--reward", reward,
            "--seed", str(seed),
            "--steps", str(args.steps),
            "--run-name", run_name,
            "--runs-dir", str(args.runs_dir),
            "--wandb-project", args.wandb_project,
            "--wandb-group", group,
            "--wandb-mode", args.wandb_mode,
            "--torch-threads", str(args.torch_threads),
            *args.extra,
        ]
        jobs.append((run_name, cmd))

    print(f"Sweep '{group}': {len(jobs)} runs, {args.workers} parallel workers "
          f"(planner={args.planner})")
    for name, _ in jobs:
        print(f"  - {name}")
    print()

    env = os.environ.copy()
    # keep BLAS/OpenMP from oversubscribing the cores across processes
    env.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
    env.setdefault("MKL_NUM_THREADS", str(args.torch_threads))
    env.setdefault("OPENBLAS_NUM_THREADS", str(args.torch_threads))

    running: list[tuple[str, subprocess.Popen]] = []
    pending = list(jobs)
    failed = []

    try:
        while pending or running:
            while pending and len(running) < args.workers:
                name, cmd = pending.pop(0)
                run_dir = args.runs_dir / name
                run_dir.mkdir(parents=True, exist_ok=True)
                log_file = open(run_dir / "log.txt", "w")
                # own process group so the terminal's Ctrl+C doesn't kill the
                # children directly — we forward a clean SIGINT ourselves below
                proc = subprocess.Popen(cmd, env=env, stdout=log_file,
                                        stderr=subprocess.STDOUT,
                                        start_new_session=True)
                running.append((name, proc))
                print(f"[started ] {name} (pid {proc.pid}) -> {run_dir / 'log.txt'}")

            time.sleep(5)
            still_running = []
            for name, proc in running:
                ret = proc.poll()
                if ret is None:
                    still_running.append((name, proc))
                elif ret == 0:
                    print(f"[finished] {name}")
                else:
                    print(f"[FAILED  ] {name} (exit {ret}) — see {args.runs_dir / name / 'log.txt'}")
                    failed.append(name)
            running = still_running

    except KeyboardInterrupt:
        print("\nInterrupted — signalling jobs to shut down cleanly "
              "(saving checkpoints + finishing wandb)...")
        # SIGINT triggers each child's `except KeyboardInterrupt` -> finally ->
        # wandb.finish(), so the runs are marked finished on the server
        for name, proc in running:
            proc.send_signal(signal.SIGINT)
        for name, proc in running:
            try:
                proc.wait(timeout=60)  # give wandb time to sync
            except subprocess.TimeoutExpired:
                print(f"  {name} did not exit in time — killing")
                proc.kill()

    print()
    print(f"Sweep '{group}' done. {len(jobs) - len(failed)}/{len(jobs)} runs succeeded.")
    if failed:
        print("Failed runs:", ", ".join(failed))


if __name__ == "__main__":
    main()
