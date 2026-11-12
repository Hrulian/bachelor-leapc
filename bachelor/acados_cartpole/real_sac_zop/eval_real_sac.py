"""Re-evaluate a saved plain-SAC checkpoint on the hardware, without any training.

Runs the same closing eval as real_sac.py (run_eval_episode: deterministic actor,
nothing goes to the buffer or wandb), but on a policy trained earlier. real_sac.py
itself cannot do that: its CHECKPOINT_DIR is the parent of the seed folders, so it
would find no checkpoint and evaluate a fresh random actor.

Only the actor is loaded and only eval_sac_run*_seed*.csv are written, next to the
checkpoint. Existing CSVs are skipped, never overwritten, so a session stopped with
Ctrl+C continues at the first missing run; delete a CSV to repeat that run.

The board has to be flashed with COMMUNICATION_TIME_MS = 50 in saz_zop.ino.

    python -m bachelor.acados_cartpole.real_sac_zop.eval_real_sac --seed 2 --runs 25
    python -m bachelor.acados_cartpole.real_sac_zop.eval_real_sac --seed 1 --dir seed1_2
"""

import argparse
import os
import queue
import threading
import time

import torch

SAC_CHECKPOINTS = os.path.join(os.path.dirname(__file__), "checkpoints", "real_sac")

# real_sac.py's max_ep_steps and STABILIZATION_BUFFER_SIZE are hard-coded for this rate
CONTROL_PERIOD_MS = 50

# opening the port resets the board, and setup() idles for over 4 s before its first frame
FIRST_FRAME_TIMEOUT_S = 20.0


def check_control_period(rs, n_frames: int = 40, rel_tol: float = 0.2):
    """Measure the Arduino's frame period and abort on a mismatch.

    Same check as real_sac_zop.py:check_control_period, which cannot be imported here
    (that module opens the same serial port). Without it, a board still flashed at
    10 ms would run the 50 ms policy at the wrong rate, in 2 s episodes.
    """
    try:
        rs.state_que.get(timeout=FIRST_FRAME_TIMEOUT_S)
    except queue.Empty:
        raise SystemExit(
            f"No frames from the Arduino within {FIRST_FRAME_TIMEOUT_S:.0f} s - is it "
            f"flashed, is the AS5600 connected, is another script holding the port?"
        )

    # flush the serial backlog, otherwise the gaps measure the listener draining it
    rs.ser.reset_input_buffer()
    time.sleep(0.3)
    with rs.state_que.mutex:
        rs.state_que.queue.clear()

    stamps = []
    try:
        for _ in range(n_frames):
            rs.state_que.get(timeout=1.0)
            stamps.append(time.monotonic())
    except queue.Empty:
        raise SystemExit("The Arduino stopped sending frames during the period check.")

    measured_ms = (stamps[-1] - stamps[0]) / (len(stamps) - 1) * 1000.0
    print(f"Measured control period: {measured_ms:.1f} ms")
    if abs(measured_ms - CONTROL_PERIOD_MS) > rel_tol * CONTROL_PERIOD_MS:
        raise SystemExit(
            f"Arduino sends every {measured_ms:.1f} ms, but the SAC policies run at "
            f"{CONTROL_PERIOD_MS} ms. Reflash COMMUNICATION_TIME_MS in saz_zop.ino."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True,
                        help="Training seed, used for the folder and the CSV names.")
    parser.add_argument("--runs", type=int, default=25,
                        help="Number of eval runs (default: 25).")
    parser.add_argument("--dir", default=None,
                        help="Checkpoint folder inside checkpoints/real_sac, for folders "
                             "not named seed<seed> (default: seed<seed>).")
    args = parser.parse_args()

    checkpoint_dir = os.path.join(SAC_CHECKPOINTS, args.dir or f"seed{args.seed}")
    actor_path = os.path.join(checkpoint_dir, "actor.pth")
    meta_path = os.path.join(checkpoint_dir, "meta.pth")
    for path in (actor_path, meta_path):
        if not os.path.exists(path):
            raise SystemExit(f"Missing checkpoint file: {path}")

    # importing real_sac opens the serial port, so only after the arguments are checked
    from bachelor.acados_cartpole.real_sac_zop import real_sac as rs

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
    # Ctrl+C while waiting for the reset aborts all remaining runs (as in real_sac.py)
    try:
        check_control_period(rs)
        for eval_run in range(1, args.runs + 1):
            csv_path = os.path.join(checkpoint_dir, f"eval_sac_run{eval_run}_seed{args.seed}.csv")
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
