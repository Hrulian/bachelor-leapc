"""Run several pure-SAC trainings (sim_sac.py) in parallel — one process per seed.

Mirrors run_parallel.py but for the MPC-free SAC baseline. By default it runs
5 seeds of the 'default' reward into a fresh wandb project so the baseline can be
compared against the SAC-ZOP sweeps side by side.

Examples:
    # 5 seeds of the default reward (the usual baseline run)
    python run_parallel_sac.py

    # compare two rewards, 5 seeds each
    python run_parallel_sac.py --rewards default energy --seeds 0 1 2 3 4
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


def parse_args():
    p = ArgumentParser(description="Parallel pure-SAC simulation sweep (baseline)")
    p.add_argument("--script", type=str, default="sim_sac.py",
                   choices=["sim_sac.py", "sim_sac_trainer.py"],
                   help="Which training script to launch per run")
    p.add_argument("--rewards", nargs="+", default=["default"], choices=sorted(REWARDS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                   help="Max concurrent processes (each uses ~1-2 threads)")
    p.add_argument("--torch-threads", type=int, default=1,
                   help="torch threads per process")
    p.add_argument("--wandb-project", type=str, default="cartpole-sac-pure-sim")
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--group", type=str, default=None,
                   help="wandb group name (default: sweep_<timestamp>)")
    p.add_argument("--runs-dir", type=Path, default=Path(__file__).parent / "runs_sac")
    # any unrecognized arguments are forwarded to sim_sac.py
    args, extra = p.parse_known_args()
    args.extra = extra
    return args


def main():
    args = parse_args()
    group = args.group or f"sweep_{int(time.time())}"
    script = Path(__file__).parent / args.script

    jobs = []
    for reward, seed in product(args.rewards, args.seeds):
        run_name = f"{group}_{reward}_s{seed}"
        cmd = [
            sys.executable, str(script),
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

    print(f"Sweep '{group}': {len(jobs)} runs, {args.workers} parallel workers")
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
        for name, proc in running:
            proc.send_signal(signal.SIGINT)
        for name, proc in running:
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                print(f"  {name} did not exit in time — killing")
                proc.kill()

    print()
    print(f"Sweep '{group}' done. {len(jobs) - len(failed)}/{len(jobs)} runs succeeded.")
    if failed:
        print("Failed runs:", ", ".join(failed))


if __name__ == "__main__":
    main()
