"""Repeated MPC evaluation episodes on the real cartpole.

The MPC baseline counterpart to `real_sac_zop.run_eval_episode`: it runs
`NUM_EVAL_RUNS` independent physical trials of the *unlearned* MPC (default
parameters, no actor, no training) and dumps one CSV + one PNG per run into a
single eval directory, plus an `eval_summary.csv` with one row per run.

CSV schema, reward (`compute_reward`) and termination rule (`done_eval`) are the
same ones real_sac_zop.py uses for its eval episodes, so the two sets of
trajectories are directly comparable - the only difference is who picks the MPC
parameters (here: nobody, they stay at their defaults). The stabilization,
performance and termination metrics mirror the ones real_sac_zop.py logs to wandb
(minus everything under `learning/`, which has no meaning without training).

Timing is specified in SECONDS (EPISODE_SECONDS, STABILIZATION_SECONDS) and
converted to env-step counts from the MEASURED control period, so a 10 ms and a
50 ms session run equally long episodes and use the same 2 s stabilization
criterion despite differing 5x in step count. The control period comes from the
Arduino firmware (COMMUNICATION_TIME_MS in saz_zop.ino), NOT from here.

Run it as a module so the package imports resolve:

    python -m bachelor.acados_cartpole.real_acados.MPC_REAL

The 10 ms vs 50 ms comparison, 20 runs each (reflash the board in between):

    TAG=dt10 python -m bachelor.acados_cartpole.real_acados.MPC_REAL
    TAG=dt50 python -m bachelor.acados_cartpole.real_acados.MPC_REAL

Both write to their own eval_logs/mpc_<planner>_<dt>ms_<tag>/ directory. Compare
the runs on `rl_return`, `episode_s`, `longest_upright_s` and `upright_fraction`
- NOT on `episodic_return` or any `*_steps` field, which scale with the rate.

Other environment variables:

    NUM_EVAL_RUNS=20       trials per session
    CONTROL_DT=auto        measured off the wire; pin with 0.010 / 0.050
    EPISODE_SECONDS=10.0   episode length
    STABILIZATION_SECONDS=2.0
    PLANNER=full  MAX_PWM=150  SHOW_PLOTS=0  EVAL_DIR=...

Ctrl+C once ends the current episode (its partial trajectory is kept, written
and plotted); Ctrl+C again during the following reset aborts the remaining runs.
"""

import csv
import os
import queue
import threading
import time
from collections import deque

import numpy as np
import serial
import torch

from bachelor.acados_cartpole.my_helpers import (
    X_TERM_M,
    compute_reward,
    counts_to_meters,
    countpersecond_to_meterspersecond,
    done_eval,
    force_to_pwm,
    sac_state_to_tensor,
)
from bachelor.acados_cartpole.my_planner_registry import (
    PLANNER_NAMES,
    make_planner,
    planner_config_dict,
)
from bachelor.acados_cartpole.my_utils_plot import plot_eval_episode_log


# EVAL CONFIGURATION ###########################################################
# Which OCP formulation is evaluated. "full" is the same planner real_sac_zop.py
# parameterizes with PLANNER=full, so this run is exactly that controller with
# default (unlearned) parameters - the baseline the SAC-ZOP numbers are read against.
PLANNER_NAME = os.environ.get("PLANNER", "full")

# Number of independent physical trials of the same controller.
NUM_EVAL_RUNS = int(os.environ.get("NUM_EVAL_RUNS", "20"))

# Application cap on |PWM|, same value the SAC-ZOP runs actuate with.
MAX_PWM = int(os.environ.get("MAX_PWM", "150"))

# Show every run's figure interactively. Off by default: plt.show() blocks, which
# would stall the hardware between runs.
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") not in ("0", "false", "False")

# TIMING #######################################################################
# Everything below is specified in SECONDS and converted to env steps once the
# control period is known. That is what makes a 10 ms and a 50 ms session
# comparable: both run 10 s episodes and both call a pole "stabilized" after 2 s
# of balancing, even though those are 1000/200 and 200/40 steps respectively.
#
# The control period is set by the Arduino FIRMWARE (COMMUNICATION_TIME_MS in
# saz_zop.ino), never here - this script is purely frame-paced. CONTROL_DT=auto
# (the default) therefore MEASURES the period off the wire instead of assuming
# it, so a mismatch between what is flashed and what the run assumes cannot
# silently produce half-length episodes. Pin it with CONTROL_DT=0.010 / 0.050.
CONTROL_DT_SETTING = os.environ.get("CONTROL_DT", "auto")

EPISODE_SECONDS = float(os.environ.get("EPISODE_SECONDS", "10.0"))
STABILIZATION_SECONDS = float(os.environ.get("STABILIZATION_SECONDS", "2.0"))
STABILIZATION_THRESHOLD = 0.15  # rad, +-0.15 rad around upright

# Window the return is binned into for `rl_return`. real_sac_zop.py makes one RL
# decision per N_STEP env steps = 50 ms and logs episode/rl_return_e as the sum of
# the per-window MEAN rewards. An env-step sum is (window / dt) x larger, so it is
# comparable neither to those curves nor between a 10 ms and a 50 ms session -
# `rl_return` is, on both counts. Kept as a literal rather than imported:
# importing real_sac_zop would open the serial port at module level.
RL_DECISION_PERIOD_S = float(os.environ.get("RL_DECISION_PERIOD_S", "0.050"))

# Derived from the control period by configure_timing(); None until then.
CONTROL_DT = None
EVAL_MAX_STEPS = None
STABILIZATION_BUFFER_SIZE = None
RL_DECISION_STEPS = None

# One directory per session, tagged with the control rate so the 10 ms and 50 ms
# sessions cannot overwrite each other. Resolved in configure_timing().
RUN_TAG = os.environ.get("TAG", time.strftime("%Y%m%d_%H%M%S"))
EVAL_DIR = None

if PLANNER_NAME not in PLANNER_NAMES:
    raise SystemExit(
        f"Unknown PLANNER '{PLANNER_NAME}'. Available: {', '.join(PLANNER_NAMES)}"
    )


# COMMUNICATION ################################################################
PORT = os.environ.get("PORT", "/dev/ttyACM0")
BAUD = 115200
# Partial-frame watchdog. Deliberately NOT derived from the control period: the
# listener thread starts before the period is known, and this is only a backstop
# (the usual recovery is the next frame's '>' terminating the stale buffer). Sized
# for the SLOWEST supported rate, which makes it safe at every faster one too.
FRAME_TIMEOUT_S = 0.05
MAX_PAYLOAD_LEN = 64
READ_TIMEOUT_S = 0.002

ser = serial.Serial(PORT, BAUD, timeout=READ_TIMEOUT_S)
time.sleep(2)
ser.reset_input_buffer()

state_que = queue.Queue()  # FIFO queue of parsed state frames
prev_sent_mode = None  # remembers the last mode sent to the Arduino for logging

# Frame width of the flashed sketch, detected from the first frame that parses:
#   5 -> saz_zop.ino:      "<x,theta,v,thetadot,tripped>", commands are "<u,mode>"
#        (has the motor-driven reset, mode 2, and reports the safety trip)
#   4 -> acados_main.ino:  "<x,theta,v,thetadot>",         commands are "<u>"
#        (no modes: `tripped` is always logged as 0 and the reset is manual)
# Detecting instead of hard-coding keeps this script usable with either sketch.
FRAME_FIELDS = None
_protocol_detected = threading.Event()

device = "cpu"


def listen_to_arduino():
    """
    -reads frames in the background with threading
    -frames come in the form of <x,theta,v,thetadot[,tripped]>\n
    -units are: <counts, rad, counts/s, rad/s, bool>
    -if a full frame got received -> put it in a que as a 5-tuple
    """

    global FRAME_FIELDS

    currently_receiving = False
    buffer = bytearray()
    t0 = None

    while True:
        b = ser.read(1)
        if not b:
            # probably something wrong with the frame
            if currently_receiving and (time.monotonic() - t0 > FRAME_TIMEOUT_S):
                currently_receiving = False
                buffer.clear()
                t0 = None
            continue

        c = b[0]

        # we are at the beginning of the payload
        if currently_receiving == False:
            if c == ord('<'):
                currently_receiving = True
                buffer.clear()
                t0 = time.monotonic()
            continue

        # we are at the end of the payload
        if c == ord('>'):
            try:
                payload = buffer.decode('ascii', errors='strict').strip()
                parts = payload.split(',')
                if len(parts) in (4, 5):
                    x, theta, v, thetadot = map(float, parts[:4])
                    # legacy sketch has no trip flag -> never tripped from the host's view
                    tripped = int(float(parts[4])) if len(parts) == 5 else 0
                    if FRAME_FIELDS is None:
                        FRAME_FIELDS = len(parts)
                        _protocol_detected.set()
                    state_que.put((x, theta, v, thetadot, tripped))
                else:
                    # malformed payload (or the legacy "ready" handshake), discard
                    pass

            except Exception:
                # decode/parsing error, discard
                pass

            currently_receiving = False
            buffer.clear()
            t0 = None
            continue

        elif c in (ord('\n'), ord('\r')):
            continue

        # we are inside the payload and buffer is smaller than max allowed length
        if len(buffer) < MAX_PAYLOAD_LEN:
            buffer.append(c)

        # overshoot of max length -> discard the reading
        else:
            currently_receiving = False
            buffer.clear()
            t0 = None


def wait_for_arduino(timeout_s: float = 20.0) -> int:
    """Block until the first frame parsed, so the wire format is known.

    Both sketches idle for 4-5 s in setup() before their first frame, and opening
    the port resets the board - hence the generous timeout.
    """
    if not _protocol_detected.wait(timeout=timeout_s):
        raise SystemExit(
            f"No valid frame on {PORT} within {timeout_s:.0f} s - is the Arduino "
            f"flashed and is another script holding the port?"
        )
    sketch = "saz_zop (modes + trip flag)" if FRAME_FIELDS == 5 else "legacy acados_main (no modes)"
    print(f"Arduino protocol: {FRAME_FIELDS}-field frames -> {sketch}")
    return FRAME_FIELDS


def measure_control_dt(n_frames: int = 60) -> float:
    """Measure the Arduino's frame period off the wire, in seconds.

    Pulls frames one at a time (no draining, so each get() returns as the listener
    pushes) and divides the TOTAL elapsed time by the number of gaps.

    That total-over-N estimator, not a median of the individual gaps: the source is
    strictly periodic (the firmware's millis() scheduler self-corrects), so all the
    per-gap scatter is host-side jitter, and it cancels in a total while it merely
    gets picked from in a median. The median version measured a true 50 ms period
    as 49 ms, which then made 10 s episodes 204 instead of 200 steps and named the
    output directory '...49ms'.

    The result is snapped to whole milliseconds, because COMMUNICATION_TIME_MS is
    an integer.

    Returns:
        The measured period in seconds.
    """
    for attempt in (1, 2):
        # Flush the OS SERIAL buffer, not just the queue. The board streams the
        # whole time the acados planner is being built at import - seconds, or
        # minutes on a rebuild - and all of that piles up in the driver buffer.
        # Clearing only state_que leaves that backlog, which the listener then
        # drains at parse speed, so the gaps measure the drain and not the frame
        # period. That is what produced a 36 ms and a 49 ms reading of the same
        # 50 ms board, and with them 13.9 s and 10.2 s "10 s" episodes.
        ser.reset_input_buffer()
        # let the listener finish any frame it is mid-way through parsing, and let
        # the wire go genuinely live, then drop whatever that produced
        time.sleep(0.3)
        with state_que.mutex:
            state_que.queue.clear()

        stamps = []
        for _ in range(n_frames):
            state_que.get(timeout=FRAME_WAIT_TIMEOUT_S)
            stamps.append(time.monotonic())

        # total-over-N, not a median: the source is strictly periodic (the
        # firmware's millis() scheduler self-corrects), so the per-gap scatter is
        # host-side jitter, which cancels in a total but merely gets picked from
        # in a median. The median read a true 50 ms as 49 ms.
        dt_raw = (stamps[-1] - stamps[0]) / (len(stamps) - 1)
        dt = round(dt_raw * 1000.0) / 1000.0

        gaps = np.diff(stamps)
        jitter = float(np.percentile(gaps, 95) - np.percentile(gaps, 5))
        print(f"Measured control period: {dt_raw*1000:.3f} ms over {len(gaps)} gaps "
              f"(p5-p95 jitter {jitter*1000:.2f} ms) -> {dt*1000:.0f} ms")

        if dt <= 0:
            raise SystemExit("Measured a non-positive frame period - is the Arduino sending?")

        # A live periodic stream jitters by a small fraction of its period; a
        # buffer draining at parse speed does not. Catching that here is what
        # stops a bad reading from silently rescaling every derived constant.
        if jitter < 0.5 * dt_raw:
            return dt

        print(f"  Reading looks unreliable (jitter {jitter*1000:.1f} ms vs period "
              f"{dt_raw*1000:.1f} ms) - probably still draining a buffered backlog."
              + ("  Retrying..." if attempt == 1 else ""))

    print("  WARNING: control period could not be measured cleanly. Pin it with "
          "CONTROL_DT=0.010 / 0.050 if the episode lengths come out wrong.")
    return dt


def configure_timing(dt: float) -> None:
    """Convert the second-based timing config into env-step counts for period `dt`."""
    global CONTROL_DT, EVAL_MAX_STEPS, STABILIZATION_BUFFER_SIZE, RL_DECISION_STEPS
    global EVAL_DIR

    CONTROL_DT = dt
    EVAL_MAX_STEPS = max(1, round(EPISODE_SECONDS / dt))
    STABILIZATION_BUFFER_SIZE = max(1, round(STABILIZATION_SECONDS / dt))
    RL_DECISION_STEPS = max(1, round(RL_DECISION_PERIOD_S / dt))

    if EVAL_DIR is None:
        EVAL_DIR = os.environ.get("EVAL_DIR") or os.path.join(
            os.path.dirname(__file__), "eval_logs",
            f"mpc_{PLANNER_NAME}_{round(dt*1000)}ms_{RUN_TAG}",
        )

    print(f"Timing @ {dt*1000:.0f} ms control period:")
    print(f"  episode:       {EPISODE_SECONDS:.1f} s  -> {EVAL_MAX_STEPS} steps")
    print(f"  stabilization: {STABILIZATION_SECONDS:.1f} s  -> {STABILIZATION_BUFFER_SIZE} steps")
    print(f"  rl_return bin: {RL_DECISION_PERIOD_S*1000:.0f} ms -> {RL_DECISION_STEPS} steps")


def talk_to_arduino(u: int, mode: int = 0) -> None:
    """
    -sends <±u,mode> + \n on the saz_zop sketch, <±u> on the legacy one
    -newline only for debugging
    """

    global prev_sent_mode
    try:
        if prev_sent_mode is None or prev_sent_mode != mode:
            print(f"[HOST MODE] mode -> {mode}")
            prev_sent_mode = mode
    except Exception:
        # be conservative: don't let logging break serial comms
        pass

    if FRAME_FIELDS == 5:
        ser.write(f"<{int(u)},{int(mode)}>\n".encode('ascii'))
    else:
        # legacy sketch drives whatever it receives, so a non-zero mode (stop /
        # reset) has to be expressed as a zero command
        ser.write(f"<{int(u) if mode == 0 else 0}>\n".encode('ascii'))


class ArduinoSilent(RuntimeError):
    """No frame arrived within the timeout - the board stopped talking."""


class ResetFailed(RuntimeError):
    """The env never reached the reset condition - usually a jammed or dead cart."""


# Give up on a reset after this long. The motor-driven reset normally takes a few
# seconds; anything past this means the cart cannot get back to centre on its own.
RESET_TIMEOUT_S = float(os.environ.get("RESET_TIMEOUT_S", "45"))


# How long to wait for a frame before giving up. Generous next to any control
# period, but finite: an unbounded state_que.get() turns a board that resets,
# crashes or gets unplugged into a process that hangs with no output at all.
FRAME_WAIT_TIMEOUT_S = 5.0


def newest_state() -> tuple:
    """Block for the next frame, drain to the freshest one, clip thetadot.

    The +-20 rad/s clip matches real_sac_zop.py: the AS5600 derivative spikes on
    fast swings, and the observation space tops out at 21.

    Raises:
        ArduinoSilent: if no frame arrives within FRAME_WAIT_TIMEOUT_S.
    """
    try:
        state = state_que.get(timeout=FRAME_WAIT_TIMEOUT_S)
    except queue.Empty:
        raise ArduinoSilent(
            f"No frame from the Arduino for {FRAME_WAIT_TIMEOUT_S:.0f} s on {PORT}. "
            f"The board reset, crashed or lost USB."
        ) from None

    while True:
        try:
            state = state_que.get_nowait()
        except queue.Empty:
            break

    x, theta, v, thetadot, tripped = state
    return (x, theta, v, float(np.clip(thetadot, -20.0, 20.0)), tripped)


def reset_env():
    """
    - brings the cart back to a defined start state (centred, still, pole hanging)
    - on the saz_zop sketch this is motor-driven (mode 2); on the legacy sketch the
      motor is stopped and the same condition is simply waited for, i.e. you push
      the cart to the centre by hand and the run starts on its own
    - drains the state queue and returns once the condition holds
    """
    if FRAME_FIELDS == 5:
        talk_to_arduino(0, mode=2)
        print("resetting env...")
    else:
        talk_to_arduino(0, mode=1)
        print("legacy sketch: no motor-driven reset - please centre the cart by "
              "hand and let the pole hang, the run starts automatically")

    # get rid of old states in a threadsafe way
    try:
        while True:
            state_que.get_nowait()
    except queue.Empty:
        pass

    last_hint = time.monotonic()
    t_start = time.monotonic()
    while True:
        x, theta, v, thetadot, tripped_flag = newest_state()

        # check reset condition:
        if (not bool(tripped_flag)                                  # not tripped
            and abs(x) <= 100                                       # be in the middle
            and abs(thetadot) <= 0.1                                # pole not moving
            and abs(countpersecond_to_meterspersecond(v)) <= 0.1    # cart not moving
            and abs(theta) >= 3.1):                                 # pole down

            break

        # else: stay in mode 2/reset until conditions are met
        waited = time.monotonic() - t_start
        if time.monotonic() - last_hint > 3.0:
            last_hint = time.monotonic()
            print(f"  waiting for reset ({waited:.0f}/{RESET_TIMEOUT_S:.0f} s): "
                  f"x={x:.0f} counts, theta={theta:.2f} rad, "
                  f"v={v:.0f} counts/s, thetadot={thetadot:.2f} rad/s, "
                  f"tripped={int(tripped_flag)}")

        # A reset that cannot complete used to spin here forever with no way out
        # but Ctrl+C - which is exactly what a jammed cart after a border crash
        # looks like, because reset_control() pushes back at PWM 25 while the
        # impact happened at PWM 150. Fail loudly instead of hanging silently.
        if waited > RESET_TIMEOUT_S:
            raise ResetFailed(
                f"Env did not reach the reset condition within {RESET_TIMEOUT_S:.0f} s.\n"
                f"  last state: x={x:.0f} counts ({counts_to_meters(x):.3f} m), "
                f"theta={theta:.2f} rad, v={v:.0f} counts/s, "
                f"thetadot={thetadot:.2f} rad/s, tripped={int(tripped_flag)}\n"
                f"  Needed: |x|<=100, |theta|>=3.1, |thetadot|<=0.1, |v|<=0.1 m/s, not tripped.\n"
                f"  If the cart is wedged at the rail end, free it by hand - the reset "
                f"drive (PWM 25) is far weaker than the crash that put it there."
            )

    print(f'reset done. State: x={x}, theta={theta}, tripped={tripped_flag}, v={v}, thetadot={thetadot}')
    return


# MPC ##########################################################################
print(f"Building planner '{PLANNER_NAME}' ...")
planner, cfg_planner = make_planner(PLANNER_NAME)
print(f"Planner '{PLANNER_NAME}' ready: {cfg_planner}")


# EVAL EPISODE #################################################################
# Per-step schema. Shared with real_sac_zop.run_eval_episode except for the two
# actor-only columns there (`param`, `actor_called`) and `theta_unwrapped` /
# `host_solve_ms` here, so a comparison script can read both by column name.
EVAL_COLUMNS = [
    'step', 'time_s', 'x_counts', 'x_m', 'theta_rad', 'theta_unwrapped',
    'v_counts_s', 'v_m_s', 'thetadot_rad_s', 'tripped', 'u_force_N', 'u_pwm',
    'reward', 'cum_reward', 'upright', 'stabilized', 'solve_success',
    'solve_retry', 'solve_time_ms', 'host_solve_ms',
]

# Counters that run ACROSS eval runs, mirroring real_sac_zop.py's `num_terminations`
# (terminations/total_e) and `num_stabilized_steps` (perf/stabilized_steps_total_e),
# which are cumulative over episodes there. Reset per session, not per run.
num_terminations = 0
num_stabilized_steps = 0


# Hysteresis for the oscillation counter [rad]. A raw sign test on theta would
# count sensor noise around upright as oscillation, and at 10 ms it would count 5x
# more of it than at 50 ms purely because it samples more often - which would make
# the metric a function of the control rate instead of the controller. Requiring
# the angle to actually travel past +-this before a swing counts kills that.
OSC_HYSTERESIS_RAD = 0.02


def _sign_changes(values, hysteresis: float) -> int:
    """Count sign changes in `values`, ignoring excursions smaller than `hysteresis`."""
    last = 0
    changes = 0
    for v in values:
        if v > hysteresis:
            s = 1
        elif v < -hysteresis:
            s = -1
        else:
            continue  # inside the deadband: not a confirmed side yet
        if last != 0 and s != last:
            changes += 1
        last = s
    return changes


def post_swingup_stats(theta_hist, thetadot_hist, u_hist, first_upright, dt) -> dict:
    """Oscillation metrics over the phase that starts at the first upright crossing.

    Three complementary views, because no single one is enough on its own:
      - theta_std      amplitude of the wobble, mean-free, so a controller that
                       simply sits at a constant angular offset is NOT punished
                       here (that is a bias, not an oscillation)
      - thetadot_rms   how much the pole actually moves; catches a fast, small
                       tremor whose positional amplitude looks harmless
      - swings_per_s   how often it crosses upright, i.e. the FREQUENCY; separates
                       a slow drift back and forth from high-frequency chatter
    u_reversals_per_s is the same idea on the actuator: a controller fighting
    itself shows up there before it shows up in the angle.

    All four are rates or spreads, never sums, so they do not scale with dt.

    One caveat on thetadot_rms: it inherits whatever noise the firmware's velocity
    estimate carries. That estimate is differentiated over STATE_UPDATE_US (5 ms)
    in saz_zop.ino, INDEPENDENTLY of COMMUNICATION_TIME_MS, so the signal is the
    same at 10 ms and at 50 ms and the RMS is comparable between them (an RMS
    converges regardless of how many samples you take of it). If STATE_UPDATE_US
    is ever tied to the frame rate, that stops being true: differentiating a noisy
    angle over a shorter window amplifies the noise by 1/dt, and the faster
    configuration would look artificially jittery.
    """
    nan = float('nan')
    if first_upright is None or first_upright >= len(theta_hist):
        return {
            'swingup_time_s': nan, 'post_swingup_s': 0.0,
            'theta_std_post': nan, 'thetadot_rms_post': nan,
            'swings_per_s': nan, 'u_reversals_per_s': nan,
        }

    th = np.asarray(theta_hist[first_upright:], dtype=float)
    td = np.asarray(thetadot_hist[first_upright:], dtype=float)
    u = np.asarray(u_hist[first_upright:], dtype=float)
    duration = len(th) * dt

    return {
        # time until the pole first reached upright - pairs with theta_abs_integral:
        # a fast swing-up scores low on both
        'swingup_time_s': round(first_upright * dt, 3),
        'post_swingup_s': round(duration, 3),
        'theta_std_post': round(float(th.std()), 5),
        'thetadot_rms_post': round(float(np.sqrt((td ** 2).mean())), 5),
        'swings_per_s': round(_sign_changes(th, OSC_HYSTERESIS_RAD) / duration, 4)
                        if duration > 0 else nan,
        'u_reversals_per_s': round(_sign_changes(u, 0.0) / duration, 4)
                             if duration > 0 else nan,
    }


def run_eval_episode(eval_run: int = 1, max_steps: int | None = None):
    """Run one MPC evaluation episode and dump the full trajectory to a CSV.

    The MPC re-solves on every incoming frame with its default parameters; there
    is no actor, no replay buffer and no wandb - the controller is fixed, this
    only measures it.

    The CSV is written incrementally (so an abort still leaves usable data) and the
    summary is appended as '#'-prefixed trailer lines, which `pandas.read_csv(...,
    comment='#')` skips.

    Args:
        eval_run: Index of this eval run (1-based), used in the CSV filename.
        max_steps: Truncation length in env steps. None -> EVAL_MAX_STEPS, i.e.
            EPISODE_SECONDS at the measured control period. Resolved here rather
            than as a default argument, which would bind before configure_timing().

    Returns:
        (csv_path, trailer_dict).
    """
    global num_terminations, num_stabilized_steps

    if max_steps is None:
        max_steps = EVAL_MAX_STEPS

    os.makedirs(EVAL_DIR, exist_ok=True)
    csv_path = os.path.join(EVAL_DIR, f"eval_mpc_{PLANNER_NAME}_run{eval_run}.csv")

    print("\n" + "=" * 60)
    print(f"MPC EVAL EPISODE {eval_run}/{NUM_EVAL_RUNS} -> {csv_path}")
    print("=" * 60)

    # bring the cart back to a defined start state, then re-arm the Arduino
    reset_env()
    talk_to_arduino(0, mode=0)

    # take the freshest state the listener has
    with state_que.mutex:
        state_que.queue.clear()
    state = newest_state()
    x, theta, v, thetadot, tripped = state

    # warm up the solver on the real state, exactly like real_sac_zop's eval does
    with torch.no_grad():
        ctx, *_ = planner(sac_state_to_tensor(state, batch=False).to(device).unsqueeze(0), ctx=None)
    print(f"Initialized MPC solver with real state: x={counts_to_meters(x):.3f}m, theta={theta:.3f}rad")

    step = 0
    cum_reward = 0.0
    max_force = 0.0
    upright_streak = 0
    longest_upright = 0
    upright_steps = 0
    stabilized = False
    stabilized_at = None
    eval_theta_buffer = deque(maxlen=STABILIZATION_BUFFER_SIZE)
    solve_times, host_times, successes = [], [], []
    terminated = False

    # return on the RL/macro-step axis: one MEAN reward per RL_DECISION_STEPS env
    # steps, summed - the same quantity real_sac_zop.py reports as rl_return_e
    rl_return = 0.0
    rl_accum = 0.0
    rl_steps_in_cycle = 0

    # |theta| accumulated over the episode (IAE w.r.t. upright). A controller that
    # swings up fast leaves the large-|theta| region sooner and therefore piles up
    # less of it; one that dawdles or falls back down keeps adding.
    theta_abs_sum = 0.0

    # kept for the post-swing-up oscillation stats. 1000 floats at 10 ms is nothing,
    # and it beats running-moment bookkeeping for something computed once at the end.
    theta_hist, thetadot_hist, u_hist = [], [], []
    first_upright = None  # index into the histories of the first upright crossing

    # how far the cart travelled at all. A cart that does not move while the
    # controller commands full force is a dead actuator, not a bad controller -
    # see episode_health_warning().
    x_min = x_max = float(x)

    # continuous angle: unwrapping makes a swing-up readable in the plot, where the
    # raw wrapped theta jumps by 2*pi every time the pole passes hanging
    theta_unwrapped = float(theta)
    theta_prev = float(theta)

    done = done_eval(state, step, max_steps, x_threshold=X_TERM_M)
    t0 = time.perf_counter()
    f = open(csv_path, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(EVAL_COLUMNS)

    try:
        while not done:
            step += 1
            obs_batch = sac_state_to_tensor(state, batch=False).to(device).unsqueeze(0)

            t_solve = time.perf_counter()
            with torch.no_grad():
                ctx, u0, _, _, _ = planner(obs_batch, ctx=ctx)
            host_ms = (time.perf_counter() - t_solve) * 1000.0
            host_times.append(host_ms)

            u_force = float(u0.detach().cpu().numpy().squeeze())
            u_pwm = force_to_pwm(
                u_force, countpersecond_to_meterspersecond(v), max_pwm_limit=MAX_PWM
            )
            max_force = max(max_force, abs(u_force))
            talk_to_arduino(u_pwm, mode=0)

            stats = ctx.log if getattr(ctx, 'log', None) else {}

            # pre-step values, paired below with the reward they produced
            row_pre = [
                step, time.perf_counter() - t0, x, counts_to_meters(x), theta,
                theta_unwrapped, v, countpersecond_to_meterspersecond(v), thetadot,
                int(tripped), u_force, u_pwm,
            ]

            # wait for the next frame and drain to the freshest one
            state = newest_state()
            x, theta, v, thetadot, tripped = state

            reward = compute_reward(state, u_force)
            cum_reward += reward

            # bin into macro-steps: close the cycle every RL_DECISION_STEPS steps
            # (and on the final, possibly short, cycle) and credit its MEAN reward
            rl_accum += reward
            rl_steps_in_cycle += 1
            if rl_steps_in_cycle == RL_DECISION_STEPS:
                rl_return += rl_accum / rl_steps_in_cycle
                rl_accum = 0.0
                rl_steps_in_cycle = 0

            # unwrap: a jump larger than pi between two frames is a wrap, not motion
            delta_theta = theta - theta_prev
            if delta_theta > np.pi:
                delta_theta -= 2 * np.pi
            elif delta_theta < -np.pi:
                delta_theta += 2 * np.pi
            theta_unwrapped += delta_theta
            theta_prev = theta

            theta_normalized = ((theta + np.pi) % (2 * np.pi)) - np.pi
            eval_theta_buffer.append(theta_normalized)
            is_upright = abs(theta_normalized) <= STABILIZATION_THRESHOLD

            x_min = min(x_min, float(x))
            x_max = max(x_max, float(x))

            # |theta| accumulator + histories for the post-swing-up oscillation stats
            theta_abs_sum += abs(theta_normalized)
            theta_hist.append(theta_normalized)
            thetadot_hist.append(thetadot)
            u_hist.append(u_force)
            if first_upright is None and is_upright:
                first_upright = len(theta_hist) - 1

            if is_upright:
                upright_streak += 1
                upright_steps += 1
                longest_upright = max(longest_upright, upright_streak)
            else:
                upright_streak = 0
            if not stabilized and len(eval_theta_buffer) == STABILIZATION_BUFFER_SIZE:
                if all(abs(t) <= STABILIZATION_THRESHOLD for t in eval_theta_buffer):
                    stabilized = True
                    stabilized_at = step

            success = float(stats.get('success_rate', float('nan')))
            retry = float(stats.get('retry_rate', float('nan')))
            solve_ms = float(stats.get('solving_time', float('nan'))) * 1000.0
            if not np.isnan(solve_ms):
                solve_times.append(solve_ms)
            if not np.isnan(success):
                successes.append(success)

            writer.writerow(row_pre + [
                reward, cum_reward, int(is_upright), int(stabilized),
                success, retry, solve_ms, host_ms,
            ])
            f.flush()  # crash/abort safe

            done = done_eval(state, step, max_steps, x_threshold=X_TERM_M)
            terminated = bool(tripped) or abs(counts_to_meters(x)) > float(X_TERM_M)

            if step % 200 == 0:
                print(f"  step {step}/{max_steps}: x={counts_to_meters(x):6.3f} m, "
                      f"theta={theta:6.3f} rad, R={cum_reward:8.2f}")

    except KeyboardInterrupt:
        print("\nEval episode aborted by user - partial trajectory kept.")
    finally:
        try:
            talk_to_arduino(0, mode=1)
        except Exception:
            pass
        elapsed = time.perf_counter() - t0

        # close a partial final cycle so its reward is not silently dropped
        if rl_steps_in_cycle:
            rl_return += rl_accum / rl_steps_in_cycle

        # cumulative session counters, same semantics as real_sac_zop.py's
        # terminations/total_e and perf/stabilized_steps_total_e (an episode in
        # which the latch ever fired contributes ALL of its steps, not just the
        # ones after it fired)
        if terminated:
            num_terminations += 1
        if stabilized:
            num_stabilized_steps += step

        trailer = {
            'controller': 'mpc',
            'planner': PLANNER_NAME,
            'eval_run': eval_run,
            # the control period this run actually ran at - the whole point of the
            # 10 ms / 50 ms comparison, and what every *_steps field below scales with
            'control_dt_s': CONTROL_DT,
            'steps': step,
            'max_steps': max_steps,
            # ---- comparable ACROSS control rates (seconds / dimensionless) -------
            'episode_s': round(step * CONTROL_DT, 3),
            # env-step reward sum: ~(50 ms / dt) x larger at 10 ms than at 50 ms, so
            # do NOT compare this between the two sessions - use rl_return
            'episodic_return': round(cum_reward, 4),
            # same return binned into fixed 50 ms windows (mean per window, summed):
            # comparable between control rates AND against real_sac_zop's rl_return_e
            'rl_return': round(rl_return, 4),
            'mean_reward_per_step': round(cum_reward / step, 4) if step else float('nan'),
            'terminated_by_trip': int(terminated),
            'stabilized': int(stabilized),
            'stabilized_at_s': (round(stabilized_at * CONTROL_DT, 3)
                                if stabilized_at is not None else None),
            'longest_upright_s': round(longest_upright * CONTROL_DT, 3),
            # total time spent upright (not necessarily consecutively)
            'upright_s': round(upright_steps * CONTROL_DT, 3),
            'upright_fraction_total': round(upright_steps / step, 4) if step else float('nan'),
            'stabilized_s': round(step * CONTROL_DT, 3) if stabilized else 0.0,
            # time integral of the reward, i.e. sum(r)*dt. The rate-independent form
            # of episodic_return: a mean/fraction cancels dt by itself, a SUM does not.
            'reward_integral': round(cum_reward * CONTROL_DT, 4),
            # IAE w.r.t. upright: integral of |theta| over the episode [rad*s].
            # LOWER IS BETTER - a fast swing-up leaves the large-|theta| region
            # sooner and stops adding to it, a slow one or one that falls back keeps
            # piling it up. Stored as the integral (sum * dt), not the raw sum, so a
            # 10 ms run is not automatically 5x "worse" than a 50 ms one.
            'theta_abs_integral': round(theta_abs_sum * CONTROL_DT, 4),
            # same thing as a plain average angle [rad] - dt-free by construction,
            # and easier to sanity-check by eye (pi/2 would be "hanging sideways
            # on average", ~0 is "upright almost the whole episode")
            'theta_abs_mean': round(theta_abs_sum / step, 5) if step else float('nan'),
            # total cart excursion over the episode - the actuator health signal
            'x_range_m': round(counts_to_meters(x_max) - counts_to_meters(x_min), 4),
            **post_swingup_stats(theta_hist, thetadot_hist, u_hist,
                                 first_upright, CONTROL_DT),
            # ---- raw step counts (rate-dependent, kept for the per-step CSV) -----
            'stabilized_at_step': stabilized_at,
            # all steps of a stabilized episode, 0 otherwise (episode/stabilized_steps_e)
            'stabilized_steps': step if stabilized else 0,
            'longest_upright_steps': longest_upright,
            # NOTE: fraction of the LONGEST unbroken upright run, not of all upright
            # steps - same definition real_sac_zop.py's eval trailer uses
            'upright_fraction': round(longest_upright / step, 4) if step else float('nan'),
            'upright_steps_total': upright_steps,
            # running totals over the whole session, not just this run
            'terminations_total': num_terminations,
            'stabilized_steps_total': num_stabilized_steps,
            'max_force_N': round(max_force, 4),
            'mean_solve_time_ms': round(float(np.mean(solve_times)), 4) if solve_times else float('nan'),
            'max_solve_time_ms': round(float(np.max(solve_times)), 4) if solve_times else float('nan'),
            'mean_host_solve_ms': round(float(np.mean(host_times)), 4) if host_times else float('nan'),
            'max_host_solve_ms': round(float(np.max(host_times)), 4) if host_times else float('nan'),
            'solver_success_rate': round(float(np.mean(successes)), 4) if successes else float('nan'),
            'wall_clock_s': round(elapsed, 3),
            'mean_step_dt_ms': round(elapsed / step * 1000.0, 3) if step else float('nan'),
        }
        for key, value in trailer.items():
            f.write(f"# {key}: {value}\n")
        f.close()

    print("-" * 60)
    for key, value in trailer.items():
        print(f"  {key}: {value}")
    print(f"Trajectory written to {csv_path}")
    print("-" * 60)
    return csv_path, trailer


# Abort the session when the rig stops responding, instead of filling the summary
# with runs that only measure a broken cart. Set HEALTH_CHECK=0 to keep going.
HEALTH_CHECK = os.environ.get("HEALTH_CHECK", "1") not in ("0", "false", "False")

# A cart that moved less than this over a whole episode did not move at all.
DEAD_CART_TRAVEL_M = 0.03
# ...and a pole this close to pi (rad) never left hanging.
DEAD_POLE_THETA_RAD = 2.8


def episode_health_warning(trailer: dict) -> str | None:
    """Return a message if this episode looks like broken hardware, else None.

    The failure mode this catches is the one that silently ruined a session: after
    a border crash the motor stopped responding, and the following runs recorded a
    pole that hangs for the full 10 s at max commanded force. Those look like
    perfectly valid rows in eval_summary.csv - same step count, same wall clock -
    but they measure the rig, not the controller.

    A bad controller moves the cart and fails to catch the pole; a dead actuator
    moves neither. Requiring BOTH conditions is what keeps a genuinely poor
    swing-up from being misreported as a hardware fault.
    """
    travel = trailer.get('x_range_m', float('nan'))
    theta_mean = trailer.get('theta_abs_mean', float('nan'))
    force = trailer.get('max_force_N', 0.0)

    if travel < DEAD_CART_TRAVEL_M and theta_mean > DEAD_POLE_THETA_RAD:
        return (f"cart moved {travel*100:.1f} cm all episode while |theta| averaged "
                f"{theta_mean:.2f} rad (pole never left hanging) at up to {force:.1f} N "
                f"commanded -> the actuator is not responding")
    return None


def write_summary(rows: list[dict]) -> str | None:
    """Rewrite eval_summary.csv from the trailers collected so far.

    Rewritten after every run rather than at the end, so an aborted session still
    leaves a complete summary of the runs that did finish.
    """
    if not rows:
        return None
    summary_path = os.path.join(EVAL_DIR, "eval_summary.csv")
    try:
        with open(summary_path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return summary_path
    except Exception as e:
        print('Failed to write summary:', e)
        return None


def print_aggregate(rows: list[dict]) -> None:
    """Print mean/std over the completed runs - the numbers that go in the table."""
    if not rows:
        print("No completed eval runs.")
        return

    def col(name):
        return np.array([r[name] for r in rows], dtype=float)

    rl_returns = col('rl_return')
    episode_s = col('episode_s')
    upright_s = col('longest_upright_s')
    upright_frac = col('upright_fraction_total')
    returns = col('episodic_return')

    print("\n" + "=" * 60)
    print(f"AGGREGATE over {len(rows)} MPC eval runs "
          f"({PLANNER_NAME}, {CONTROL_DT*1000:.0f} ms control period)")
    print("=" * 60)
    print("  -- comparable across control rates --")
    print(f"  rl_return:             {rl_returns.mean():8.3f} +- {rl_returns.std():.3f}")
    print(f"  episode_s:             {episode_s.mean():8.3f} +- {episode_s.std():.3f}  s")
    print(f"  longest_upright_s:     {upright_s.mean():8.3f} +- {upright_s.std():.3f}  s")
    print(f"  upright_fraction:      {upright_frac.mean():8.3f} +- {upright_frac.std():.3f}")
    print(f"  stabilized:            {sum(r['stabilized'] for r in rows):8d}/{len(rows)} runs")
    print(f"  terminated_by_trip:    {sum(r['terminated_by_trip'] for r in rows):8d}/{len(rows)} runs")
    print("  -- swing-up speed (lower is better) --")
    print(f"  theta_abs_integral:    {col('theta_abs_integral').mean():8.3f} +- {col('theta_abs_integral').std():.3f}  rad*s")
    print(f"  theta_abs_mean:        {col('theta_abs_mean').mean():8.4f} +- {col('theta_abs_mean').std():.4f}  rad")
    print(f"  swingup_time_s:        {np.nanmean(col('swingup_time_s')):8.3f} +- {np.nanstd(col('swingup_time_s')):.3f}  s")
    print("  -- oscillation after swing-up (lower is steadier) --")
    print(f"  theta_std_post:        {np.nanmean(col('theta_std_post')):8.4f} +- {np.nanstd(col('theta_std_post')):.4f}  rad")
    print(f"  thetadot_rms_post:     {np.nanmean(col('thetadot_rms_post')):8.4f} +- {np.nanstd(col('thetadot_rms_post')):.4f}  rad/s")
    print(f"  swings_per_s:          {np.nanmean(col('swings_per_s')):8.3f} +- {np.nanstd(col('swings_per_s')):.3f}  1/s")
    print(f"  u_reversals_per_s:     {np.nanmean(col('u_reversals_per_s')):8.3f} +- {np.nanstd(col('u_reversals_per_s')):.3f}  1/s")
    print("  -- rate-dependent (do not compare 10 ms vs 50 ms) --")
    print(f"  episodic_return:       {returns.mean():8.3f} +- {returns.std():.3f}  (env-step sum)")
    print(f"  steps:                 {col('steps').mean():8.1f} +- {col('steps').std():.1f}")
    print("=" * 60)


def main():
    print(f"Config: {planner_config_dict(PLANNER_NAME, cfg_planner)}")

    # start the background receiver task
    arduinoThread = threading.Thread(target=listen_to_arduino, args=())
    arduinoThread.daemon = True
    arduinoThread.start()

    # find out which sketch is on the board before sending the first command
    wait_for_arduino()

    # the firmware owns the control period, so measure it rather than assume it -
    # otherwise a board still flashed at 50 ms would silently turn a "10 ms, 10 s"
    # session into 200 steps = 2 s episodes with a 0.4 s stabilization criterion
    if CONTROL_DT_SETTING == "auto":
        dt = measure_control_dt()
    else:
        dt = float(CONTROL_DT_SETTING)
        measured = measure_control_dt()
        if abs(measured - dt) > 0.002:
            print(f"\n  WARNING: CONTROL_DT={dt*1000:.0f} ms was pinned, but the board "
                  f"is sending every {measured*1000:.1f} ms.\n"
                  f"  Episode lengths and the stabilization window will be wrong. "
                  f"Reflash COMMUNICATION_TIME_MS in saz_zop.ino or drop CONTROL_DT.\n")

    configure_timing(dt)
    os.makedirs(EVAL_DIR, exist_ok=True)
    print(f"Eval directory: {EVAL_DIR}")

    summary_rows: list[dict] = []

    try:
        for eval_run in range(1, NUM_EVAL_RUNS + 1):
            print(f"\n>>> Eval run {eval_run}/{NUM_EVAL_RUNS}")
            csv_path, trailer = run_eval_episode(eval_run=eval_run)
            summary_rows.append(trailer)

            # one figure per run, saved next to its CSV
            try:
                plot_eval_episode_log(
                    csv_path,
                    plt_show=SHOW_PLOTS,
                    title=f"MPC ({PLANNER_NAME}) - eval run {eval_run}",
                    x_limit=float(X_TERM_M),
                    upright_threshold=STABILIZATION_THRESHOLD,
                )
            except Exception as e:
                print('Plotting failed:', e)

            write_summary(summary_rows)

            # Cross-check the startup measurement against the period the episode
            # actually ran at. These must agree; when they do not, every derived
            # constant is off by their ratio - episode length, the stabilization
            # window and all *_s metrics - and the CSVs look perfectly normal
            # while being silently mis-scaled.
            actual = trailer.get('mean_step_dt_ms', float('nan'))
            assumed = CONTROL_DT * 1000.0
            if actual == actual and abs(actual - assumed) / assumed > 0.05:
                print(f"\n  WARNING: run {eval_run} ran at {actual:.1f} ms/step but the "
                      f"session assumes {assumed:.0f} ms.")
                print(f"  Episodes are {EPISODE_SECONDS * actual / assumed:.1f} s, not "
                      f"{EPISODE_SECONDS:.1f} s, and every *_s metric is off by "
                      f"{actual / assumed:.2f}x.")
                print(f"  Restart with CONTROL_DT={round(actual)/1000:.3f} to pin it.")

            # stop the session rather than logging more runs of a broken rig
            problem = episode_health_warning(trailer) if HEALTH_CHECK else None
            if problem:
                print("\n" + "!" * 60)
                print(f"HARDWARE FAULT after run {eval_run}: {problem}.")
                print("Aborting the session - further runs would only record this.")
                print("Free the cart / power-cycle the motor driver, then restart.")
                print(f"Runs 1-{eval_run - 1} are intact; this run is NOT usable.")
                print("!" * 60)
                break

    except KeyboardInterrupt:
        print("\nRemaining eval runs skipped.")
    except (ArduinoSilent, ResetFailed) as e:
        # expected, self-explaining failures: no traceback needed, and the runs
        # already completed stay in the summary written by the finally block
        print(f"\n{'!' * 60}\n{e}\n{'!' * 60}")
    except Exception as e:
        import traceback
        print('Eval run failed:', e)
        traceback.print_exc()
    finally:
        try:
            talk_to_arduino(0, mode=1)  # tell arduino to stop
        except Exception:
            pass
        summary_path = write_summary(summary_rows)
        print_aggregate(summary_rows)
        if summary_path:
            print(f"Summary written to {summary_path}")
        ser.close()


if __name__ == "__main__":
    main()
