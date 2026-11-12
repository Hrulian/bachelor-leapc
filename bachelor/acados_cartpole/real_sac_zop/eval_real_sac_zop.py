"""Re-evaluate a saved 50 ms SAC-ZOP checkpoint on the hardware, without any training.

The SAC-ZOP counterpart of eval_real_sac.py. It runs the same closing eval as
real_sac_zop.py (run_eval_episode: deterministic actor, nothing goes to the buffer
or wandb), but on a policy trained earlier. real_sac_zop.py itself cannot do that:
it would train first, and its CHECKPOINT_DIR is the parent of the seed folders.

Only the actor is loaded and only eval_<planner>_run*_seed*.csv are written, next to
the checkpoint. Existing CSVs are skipped, never overwritten, so a session stopped
with Ctrl+C continues at the first missing run; delete a CSV to repeat that run.
--runs is the last run number, not the number of new runs: with runs 1-5 on disk,
--runs 30 adds 25.

The board has to be flashed with COMMUNICATION_TIME_MS = 50 in saz_zop.ino.

    python -m bachelor.acados_cartpole.real_sac_zop.eval_real_sac_zop --seed 0 --runs 30
    python -m bachelor.acados_cartpole.real_sac_zop.eval_real_sac_zop --planner cart --seed 5
"""

import argparse
import os
import threading

import torch

from bachelor.acados_cartpole.real_sac_zop.eval_real_sac import (
    CONTROL_PERIOD_MS,
    check_control_period,
)

CHECKPOINTS = os.path.join(os.path.dirname(__file__), "checkpoints")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--planner", default="fullcart",
                        help="OCP formulation of the checkpoint, fullcart = ZOP-Pos "
                             "(default: fullcart).")
    parser.add_argument("--seed", type=int, required=True,
                        help="Training seed, used for the folder and the CSV names.")
    parser.add_argument("--runs", type=int, default=25,
                        help="Evaluate up to this run number (default: 25).")
    parser.add_argument("--dir", default=None,
                        help="Checkpoint folder inside checkpoints/real_sac_zop_<planner>, "
                             "for folders not named s<seed> (default: s<seed>).")
    args = parser.parse_args()

    checkpoint_dir = os.path.join(
        CHECKPOINTS, f"real_sac_zop_{args.planner}", args.dir or f"s{args.seed}"
    )
    actor_path = os.path.join(checkpoint_dir, "actor.pth")
    meta_path = os.path.join(checkpoint_dir, "meta.pth")
    for path in (actor_path, meta_path):
        if not os.path.exists(path):
            raise SystemExit(f"Missing checkpoint file: {path}")

    # real_sac_zop reads these on import and sizes the planner and the episode from them
    os.environ["PLANNER"] = args.planner
    os.environ["CONTROL_PERIOD_MS"] = str(CONTROL_PERIOD_MS)
    os.environ["N_STEP"] = "1"

    # importing real_sac_zop opens the serial port, so only after the arguments are checked
    from bachelor.acados_cartpole.real_sac_zop import real_sac_zop as rs

    # not rs.load_checkpoints(): on a failed load it carries on with the random actor
    rs.actor.load_state_dict(torch.load(actor_path, map_location=rs.device))
    meta = torch.load(meta_path, map_location=rs.device)
    rs.episode_count = int(meta["episode_count"])
    rs.abs_step_count = int(meta["abs_step_count"])
    print(f"Loaded actor from {actor_path} (episode_count={rs.episode_count}, "
          f"abs_step_count={rs.abs_step_count}, wandb_run_id={meta.get('wandb_run_id')})")

    # run_eval_episode writes its CSV here
    rs.CHECKPOINT_DIR = checkpoint_dir

    arduinoThread = threading.Thread(target=rs.listen_to_arduino, args=())
    arduinoThread.daemon = True
    arduinoThread.start()

    # Ctrl+C during an episode keeps its partial CSV and goes on with the next run,
    # Ctrl+C while waiting for the reset aborts all remaining runs (as in real_sac_zop.py)
    try:
        check_control_period(rs)
        for eval_run in range(1, args.runs + 1):
            csv_path = os.path.join(
                checkpoint_dir, f"eval_{args.planner}_run{eval_run}_seed{args.seed}.csv"
            )
            if os.path.exists(csv_path):
                print(f"Eval run {eval_run}/{args.runs}: {csv_path} exists, skipping")
                continue
            print(f"\n>>> Eval run {eval_run}/{args.runs} (seed={args.seed})")
            rs.run_eval_episode(eval_run=eval_run, seed=args.seed)
    except KeyboardInterrupt:
        print("Remaining eval runs skipped.")
    finally:
        try:
            rs.talk_to_arduino(0, mode=1)
        except Exception:
            pass
        rs.ser.close()


if __name__ == "__main__":
    main()
