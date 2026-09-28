
import torch, threading, queue, serial, time, os, csv
import numpy as np
import gymnasium as gym
import wandb
from collections import deque
from bachelor.acados_cartpole.real_sac_zop.my_sac import SacCritic
from bachelor.acados_cartpole.real_sac_zop.my_sac_zop import  MpcSacActor, SacZopTrainerConfig
from bachelor.acados_cartpole.real_sac_zop.my_utils import soft_target_update
from bachelor.acados_cartpole.real_sac_zop.my_buffer import ReplayBuffer
from bachelor.acados_cartpole.my_planner_registry import (
    make_planner,
    planner_config_dict,
    PLANNER_NAMES,
)
from bachelor.acados_cartpole.my_helpers import (
    force_to_pwm,
    countpersecond_to_meterspersecond,
    counts_to_meters,
    compute_reward,
    done_eval,
    sac_state_to_tensor,
    X_TERM_M,
)
from bachelor.acados_cartpole.my_utils_plot import plot_policy_heatmap

 
from leap_c.planner import ControllerFromPlanner
from leap_c.torch.nn.extractor import get_extractor_cls  # optional helper



#PLANNER SELECTION#################################################################
# Which OCP formulation the SAC-ZOP actor parameterizes. See my_planner_registry.py
# for the list; change the default here or override per run without editing the file:
#
#     PLANNER=cart python -m bachelor.acados_cartpole.real_sac_zop.real_sac_zop
#
# Each variant exposes a different param_space, so the critic/actor shapes differ
# and checkpoints are NOT interchangeable -> CHECKPOINT_DIR is per planner below.
PLANNER_NAME = os.environ.get("PLANNER", "full")

# Optional per-variant config overrides, e.g. {"tunable": {"M": 0.17444 * 8}} to
# run the model-mismatch ablation. Empty = use each variant's own defaults.
PLANNER_OVERRIDES: dict = {}

# Run one deterministic, unlogged evaluation episode on Ctrl+C (see run_eval_episode).
# Set to False (or PLANNER=... EVAL_ON_EXIT=0) to exit immediately instead.
RUN_EVAL_ON_EXIT = os.environ.get("EVAL_ON_EXIT", "1") not in ("0", "false", "False")

if PLANNER_NAME not in PLANNER_NAMES:
    raise SystemExit(
        f"Unknown PLANNER '{PLANNER_NAME}'. Available: {', '.join(PLANNER_NAMES)}"
    )


#COMMUNICATION#####################################################################
# wandb communication parameters
os.environ['WANDB_API_KEY'] = 'fd053eb0471b83f999819cd4c4e4930ea28de0ea'

# serial communication parameters
PORT = "/dev/ttyACM0"
BAUD = 115200
FRAME_TIMEOUT_S = 0.01   # partial-frame watchdog: ~1 control period (10 ms), still
                         # ~3.5x the ~2.8 ms a frame needs on the wire at 115200 baud
MAX_PAYLOAD_LEN = 64
READ_TIMEOUT_S = 0.002

ser = serial.Serial(PORT, BAUD, timeout=READ_TIMEOUT_S)
time.sleep(2)
ser.reset_input_buffer()

state_que = queue.Queue() # init que that stores states
prev_sent_mode = None  # remembers last mode sent to Arduino for logging

# Tag for the Arduino control period this script's constants assume (see
# COMMUNICATION_TIME_MS in saz_zop.ino, FRAME_TIMEOUT_S above, max_ep_steps,
# STABILIZATION_BUFFER_SIZE and train_start below - all hard-coded for 10 ms,
# unlike real_acados/MPC_REAL.py, which measures the rate off the wire). Folded
# into CHECKPOINT_DIR so a 50 ms checkpoint/actor/critic set is never silently
# loaded into a 10 ms run (shape-compatible, but trained on a different timescale
# - see N_STEP below) or overwritten by one.
CONTROL_RATE_TAG = "10ms"

# checkpoint directory for models - per planner AND per control rate, because the
# param_space (and with it the critic/actor input shapes) differs between OCP
# formulations, while the control rate changes what the SAME shapes were trained
# on. Sharing a directory across either would make load_checkpoints() either fail
# on a shape mismatch or silently resume into the wrong timescale.
CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), "checkpoints",
    f"real_sac_zop_{PLANNER_NAME}_{CONTROL_RATE_TAG}",
)

# timing tracking for training step
training_step_times = []  # list to store duration of each training call


def listen_to_arduino():
    """
    -reads frames in the background with threading
    -frames come in the form  of <x,theta,v,thetadot,tripped>\n
    -units are: <counts, rad, counts/s, rad/s, bool>
    -if full frame got received -> put it inn a que as tuple
    """
    
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
                if len(parts) == 5:
                    # payload format: <x,theta,v,thetadot,tripped>
                    x, theta, v, thetadot = map(float, parts[:4])
                    tripped = int(parts[4])
                    # store in expected order for controller: x, theta, v, thetadot, tripped
                    state_que.put((x, theta, v, thetadot, tripped))
                else:
                    # malformed payload, discard
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
            

def talk_to_arduino(u: int, mode: int) -> None :
    """
    -sends <±u,mode> + \n 
    -newline only for debugging
    """
    
    global prev_sent_mode
    # print mode change only when it differs from last sent mode
    try:
        if prev_sent_mode is None or prev_sent_mode != mode:
            print(f"[HOST MODE] mode -> {mode}")
            prev_sent_mode = mode
    except Exception:
        # be conservative: don't let logging break serial comms
        pass

    ser.write(f"<{u},{mode}>\n".encode('ascii'))



#LEARNING###############################################################################
# counting semaphore to notify learner thread of new data
# (threading.Event is binary and loses signals fired during a training block;
# a Semaphore counts every release() so no transition is silently dropped)
_learning_sem = threading.Semaphore(0)

# drain synchronization: lets inbetween_training() block until the learner thread
# has caught up on every token released so far (no concurrent updates needed).
_drain_cond = threading.Condition()
_tokens_released = 0   # total _learning_sem.release() calls
_tokens_processed = 0  # tokens the learner thread has consumed + handled


def _note_token_released():
    """Release one learner token and count it (for drain bookkeeping)."""
    global _tokens_released
    with _drain_cond:
        _tokens_released += 1
    _learning_sem.release()


def _note_token_processed():
    """Mark one acquired token as fully handled and wake any drain waiter."""
    global _tokens_processed
    with _drain_cond:
        _tokens_processed += 1
        _drain_cond.notify_all()


def _safe_mean(values):
    """Mean of a list, returning nan on empty instead of warning."""
    return float(np.mean(values)) if values else float('nan')


# initialize counters
episode_step_count = 0 # steps in current episode
total_training_steps = 0  # einzelne training steps (critic + actor updates) - für soft_update_freq
learning_step = 0   # blocks (20-step blocks) - für logging und tracking
episode_count = 0   # total episodes completed
abs_step_count = 0  # total ENV steps (Arduino frames) across all episodes
# total RL steps (= completed N-step cycles) across all episodes. THIS is the wandb
# step axis - see the naming-convention block below for why.
rl_step_count = 0

# episode reward tracking
episode_rewards = []  # list of cumulative rewards per episode
current_episode_reward = 0.0  # accumulated reward in current episode
max_force_perep = 0  # track max force per episode
wandb_run_id = None  # wandb run ID for resuming runs

# metrics mirrored from the sim scripts (sim_sac / sim_sac_zop / sim_sac_zopfill)
num_terminations = 0  # cumulative episodes that ended in a trip (terminated, not truncated)
num_stabilized_steps = 0  # cumulative steps spent in the balanced/stabilized mode

# stabilization tracking
STABILIZATION_BUFFER_SIZE = 200  # 2 s of balancing at the 10 ms control rate
STABILIZATION_THRESHOLD = 0.15  # rad, ±0.15 rad around upright
theta_buffer = deque(maxlen=STABILIZATION_BUFFER_SIZE)  # FIFO queue for theta values
stabilized_this_episode = False  # flag: was pole stabilized this episode
stabilized_at_step = None  # episode step at which the latch fired, None if it never did

# longest run of consecutive steps within STABILIZATION_THRESHOLD in the current episode.
# Same quantity the latch thresholds, but reported ungated: rotation tops out at a few
# steps while real balancing reaches the full episode, so it separates "hopeless" from
# "almost there" where the binary flag stays stuck at 0.
upright_streak = 0
longest_upright_steps = 0

# THE WANDB STEP AXIS IS THE RL STEP, NOT THE ENV STEP.
#
# One RL step = one N_STEP cycle = one actor decision = one buffer transition.
# At 10 ms with N_STEP=5 that is 50 ms of wall clock; at the old 50 ms rate with
# N_STEP=1 it was also 50 ms - and there abs_step_count and rl_step_count were the
# same number, which is exactly why this axis makes the two sets of runs overlay.
# Logging on abs_step_count instead would stretch every 10 ms curve 5x along x and
# make it incomparable with everything recorded before.
#
# Consequence: there is ONE wandb row per cycle, not per env step. Per-step
# quantities are therefore folded into the cycle:
#   - state / action  -> the value at the cycle start, i.e. what the ACTOR saw and
#                        chose. At N=1 this is identical to the old per-step value.
#   - reward          -> the cycle MEAN (= the macro reward that enters the buffer),
#                        which is what a single 50 ms sample estimated before.
#   - upright         -> the FRACTION of the cycle spent upright. At N=1 that is
#                        the same 0/1 signal as before, so averaging it in the
#                        wandb UI still gives "fraction of time upright".
#   - solver stats    -> mean over the cycle, so the N-1 held steps are represented
#                        too rather than only the actor step.
#
# The metric NAMES are unchanged from the pre-K-step runs on purpose: same keys,
# same panels, just folded onto the RL axis. Nothing new is introduced, so an old
# and a new run can be dropped into one wandb chart without touching the config.
#
# Naming convention for every logged metric:
#   *_s  -> one value per RL step, meant to be plotted over the step axis
#   *_e  -> one value per episode, meant to be plotted over episode/episode_number_e
# Everything goes to wandb at step=rl_step_count (wandb has a single step axis);
# the suffix says which x-axis the metric is meant to be read on.

# max steps per episode (env steps, i.e. Arduino frames - NOT RL decisions)
max_ep_steps = 1000  # 10 s at the 10 ms control rate

# N-step (zero-order-hold) cycle length: the actor picks an MPC parameter every
# N env steps, the MPC re-solves every step with that HELD param, and ONE
# accumulated-reward transition per cycle goes into the buffer. N=1 degenerates to
# "actor every step". Consequences of N>1, all of them intended:
#   - update-to-data ratio: one learner token per cycle, so 20 gradient steps per
#     N env steps (N=5 -> 4 per env step instead of 20). To keep real_sac.py
#     comparable, set its LEARN_EVERY to the same N.
#   - the buffer holds macro-transitions (obs at cycle start, held param, MEAN
#     reward over the cycle, obs at cycle end). The critic target bootstraps with
#     plain `gamma`, i.e. gamma discounts per MACRO-step, not per env step.
#     NOTE: simulation_zaczop/sim_sac_zop.py sums instead of averaging in its
#     --K_step path; that is a constant factor N, so the optimal policy is the
#     same, but its Q values run N x larger than the ones here.
#
# N x control period = the RL decision period, and THAT is what gamma discounts.
# At the 10 ms control rate, N=5 keeps the macro-step at 50 ms, i.e. exactly the
# RL rate of the earlier 50 ms / N=1 runs: same effective horizon at gamma=0.99,
# same tokens/s, same gradient-steps/s, and - because the macro reward is the mean
# rather than the sum - the same reward scale. What actually changes is that the
# MPC re-solves 5x per hold instead of once, which is the point of the exercise.
N_STEP = 5

# device setup
device = "cpu"
# MPC Layer Setup - the variant is chosen by PLANNER_NAME at the top of the file
print(f"Building planner '{PLANNER_NAME}' ...")
planner, cfg_planner = make_planner(
    PLANNER_NAME, **PLANNER_OVERRIDES.get(PLANNER_NAME, {})
)
controller_wrapped = ControllerFromPlanner(planner)
print(f"Planner '{PLANNER_NAME}' ready: {cfg_planner}")
ctx = None


# observation and action spaces
# state is (x, theta, xdot, thetadot)
# use planner config for reasonable x-bounds and clamp angle to [-pi, pi]
# episode termination threshold - deliberately NOT cfg_planner.x_threshold: that value is
# the MPC's own hard box constraint on x (my_acados_ocp.py), a controller design choice.
# Termination is an environment property and is shared with real_sac.py via X_TERM_M.
_x_thr = X_TERM_M
_x_low = -float(_x_thr) # in meters
_x_high = float(_x_thr) # in meters

#TODO: double check if these spaces are correct 
obs_low = np.array([_x_low, -np.pi, -5, -21], dtype=np.float32)
obs_high = np.array([_x_high, np.pi, 5, 21], dtype=np.float32)
obs_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

action_space = controller_wrapped.param_space


# SacZop config
cfg_saczop = SacZopTrainerConfig()

# Enable layer normalization for critic only
cfg_saczop.critic_mlp.norm_layer = "layer_norm"

# collect this many env steps before the first gradient step (same as real_sac.py).
# With train_start=0 the first updates run on a handful of near-identical transitions
# from a single episode start, which the critic overfits hard at this UTD.
# 1000 env steps = 10 s of hardware data at the 10 ms control rate = 200 buffered
# N-step transitions, comfortably above batch_size (64), so the first block after
# the gate actually trains instead of running 20 no-ops.
cfg_saczop.train_start = 1000

# Replay Buffer init
replay_buffer = ReplayBuffer(buffer_limit=cfg_saczop.buffer_size, device=device)


# critic init
extractor_cls = get_extractor_cls("identity")

critic = SacCritic(
    extractor_cls=extractor_cls,
    observation_space=obs_space,
    action_space=action_space,
    mlp_cfg=cfg_saczop.critic_mlp,
    num_critics=cfg_saczop.num_critics,
).to(device)

target_critic = SacCritic(
    extractor_cls=extractor_cls,
    observation_space=obs_space,
    action_space=action_space,
    mlp_cfg=cfg_saczop.critic_mlp,
    num_critics=cfg_saczop.num_critics,
).to(device)

target_critic.load_state_dict(critic.state_dict())


# actor init
actor = MpcSacActor(
    extractor_cls=extractor_cls,
    observation_space=obs_space,
    controller=controller_wrapped,
    distribution_name=cfg_saczop.distribution_name,
    mlp_cfg=cfg_saczop.actor_mlp,
    init_param_with_default=cfg_saczop.init_param_with_default,
).to(device)


# entropy temperature ALpha init
log_alpha = torch.nn.Parameter(
    torch.tensor(cfg_saczop.init_alpha, dtype=torch.float32).log()
).to(device)


alpha_optimizer = (
    torch.optim.Adam([log_alpha], lr=cfg_saczop.lr_alpha)
    if cfg_saczop.lr_alpha is not None
    else None
)

param_dim = int(np.prod(action_space.shape))
action_dim = 1
entropy_norm = param_dim / action_dim
# standard SAC heuristic for a 1-D action (= -1.0), matching real_sac.py and
# sim_sac_zop.py rather than the -2.0 in SacTrainerConfig
target_entropy = -float(action_dim)


# initializing optimizers
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg_saczop.lr_q)
actor_optimizer  = torch.optim.Adam(actor.parameters(),  lr=cfg_saczop.lr_pi)



def sac_zop_update_step(batch_size, train_start):
    """
    Background thread: one training block (20 gradient steps) per buffer drop.
    1 semaphore token = 1 block. If tokens pile up (training slower than sampling)
    the thread runs back-to-back without waiting, using all available time.
    """
    global abs_step_count, learning_step, total_training_steps, rl_step_count
    timeout_s = 1.0
    actor_update_freq = 20
    LOG_FREQ = 10

    while True:
        try:
            if not _learning_sem.acquire(timeout=timeout_s):
                continue

            # token consumed: mark it processed once handled (also on the warmup
            # skip path) so inbetween_training()'s drain can never hang.
            try:
                if abs_step_count < train_start:
                    continue

                block_start_time = time.perf_counter()

                # only pay .item() cost on blocks we'll actually log
                will_log = (learning_step + 1) % LOG_FREQ == 0
                metrics_accumulator = {
                    'q_losses': [], 'pi_losses': [], 'alphas': [],
                    'q_values': [], 'q_targets': [], 'entropies': [],
                } if will_log else None

                for i in range(20):
                    saczop_single_step_update(
                        batch_size,
                        update_actor=(i % actor_update_freq == 0),
                        metrics_accumulator=metrics_accumulator,
                    )

                learning_step += 1
            finally:
                _note_token_processed()

            block_elapsed_time = time.perf_counter() - block_start_time
            training_step_times.append(block_elapsed_time)

            if learning_step % LOG_FREQ == 0:
                try:
                    wandb.log({
                        'learning/training_block_time_ms_s': block_elapsed_time * 1000,
                        'learning/block_step_s': learning_step,
                        'learning/total_training_steps_s': total_training_steps,
                        # backlog: tokens released but not yet processed (lock-free read)
                        'learning/pending_tokens_s': _tokens_released - _tokens_processed,
                        'learning/buffer_size_s': len(replay_buffer),
                        'learning/q_loss_avg_s': _safe_mean(metrics_accumulator['q_losses']),
                        'learning/pi_loss_s': metrics_accumulator['pi_losses'][0] if metrics_accumulator['pi_losses'] else float('nan'),
                        'learning/alpha_s': metrics_accumulator['alphas'][0] if metrics_accumulator['alphas'] else float('nan'),
                        'learning/q_avg_s': _safe_mean(metrics_accumulator['q_values']),
                        'learning/q_target_avg_s': _safe_mean(metrics_accumulator['q_targets']),
                        'learning/entropy_s': metrics_accumulator['entropies'][0] if metrics_accumulator['entropies'] else float('nan'),
                    }, step=rl_step_count)
                except Exception:
                    pass

        except Exception as e:
            print("Exception in sac_zop_update_step:", e)


def saczop_single_step_update(batch_size, update_actor=True, metrics_accumulator=None):
    """
    Pure training step: Performs one critic+actor update.
    Increments total_training_steps counter.
    Only collects metrics into dict - NO logging.
    
    Args:
        batch_size: Number of samples to use.
        update_actor: Whether to update actor this step.
        metrics_accumulator: Dict to collect metrics (required, not optional).
    """
    
    global total_training_steps

    if len(replay_buffer) < batch_size:
        return False

    o, a, r, o_prime, te = replay_buffer.sample(batch_size)

    # cache alpha once — avoids repeated exp() + .item() syncs
    alpha = log_alpha.exp().item()

    # critic target (no grad)
    with torch.no_grad():
        pi_o_prime = actor(o_prime, None, only_param=True)
        q_target = target_critic(o_prime, pi_o_prime.param)
        q_target = torch.min(q_target, dim=1, keepdim=True).values
        factor = cfg_saczop.entropy_reward_bonus / entropy_norm
        q_target = q_target - alpha * pi_o_prime.log_prob * factor
        target = r[:, None].to(device) + cfg_saczop.gamma * (1 - te[:, None].to(device)) * q_target

    # actor forward + alpha update BEFORE critic — only on actor steps
    if update_actor:
        pi_o = actor(o, None, only_param=True)
        a_pi = pi_o.param
        log_p = pi_o.log_prob / entropy_norm

        if alpha_optimizer is not None:
            alpha_loss = -torch.mean(log_alpha.exp() * (log_p + target_entropy).detach())
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

    # critic update (always)
    q = critic(o, a)
    q_loss = torch.mean((q - target).pow(2))
    critic_optimizer.zero_grad()
    q_loss.backward()
    critic_optimizer.step()

    # actor update — only on actor steps
    if update_actor:
        q_pi = critic(o, a_pi)
        min_q_pi = torch.min(q_pi, dim=1, keepdim=True).values
        pi_loss = (alpha * log_p - min_q_pi).mean()
        actor_optimizer.zero_grad()
        pi_loss.backward()
        actor_optimizer.step()

    if total_training_steps % cfg_saczop.soft_update_freq == 0:
        soft_target_update(critic, target_critic, cfg_saczop.tau)

    total_training_steps += 1

    if metrics_accumulator is not None:
        metrics_accumulator['q_losses'].append(q_loss.item())
        metrics_accumulator['q_values'].append(q.mean().item())
        metrics_accumulator['q_targets'].append(target.mean().item())
        if update_actor:
            metrics_accumulator['pi_losses'].append(pi_loss.item())
            metrics_accumulator['alphas'].append(alpha)  # already float, no .item() needed
            metrics_accumulator['entropies'].append(-log_p.mean().item())

    return True


def inbetween_training(timeout_s: float | None = 30.0):
    """
    Block until the background learner thread has drained all tokens released so
    far, i.e. every transition queued up to this point has been trained on.

    Does NOT run gradient updates itself — the learner thread owns all updates,
    so there is no concurrent in-place modification of the networks (which used
    to trigger the autograd version-counter RuntimeError between episodes).

    Args:
        timeout_s: Safety cap in seconds. None blocks indefinitely.

    Returns:
        True if fully drained, False if the timeout was hit first.
    """
    with _drain_cond:
        target = _tokens_released  # snapshot: everything queued up to now
        drained = _drain_cond.wait_for(
            lambda: _tokens_processed >= target, timeout=timeout_s
        )
    if not drained:
        print(f"inbetween_training: drain timed out "
              f"({_tokens_processed}/{target} tokens processed)")
    return drained


def reset_env():
    """
    - sends reset command to arduino
    - waits until physical env is reset (cart near center, low velocity, not tripped)
    - drains state queue to freshest state
    - returns nothing
    """
    # send reset command to arduino: mode 2 = reset
    talk_to_arduino(0, mode=2)
    print("resetting env...")

    # get rid of old states in a threadsafe way
    try:
        while True:
            state_que.get_nowait()
    except queue.Empty:
        pass

    while True:
        # block for next available state
        state = state_que.get()

        # drain to the most recent state
        while True:
            try:
                state = state_que.get_nowait()
            except queue.Empty:
                break

        # expect state = (x, theta, v, thetadot, tripped_flag)
        if len(state) < 5:
            continue

        x, theta, v, thetadot, tripped_flag = state

        # check reset condition:
        if (not bool(tripped_flag)                                  # not tripped
            and abs(x) <= 100                                       # be in the middle
            and abs(thetadot) <= 0.1                             # pole not moving
            and abs(countpersecond_to_meterspersecond(v)) <= 0.1    # cart not moving
            and abs(theta) >= 3.1):                                # pole down
                
            break
        # else: stay in mode 2/reset until conditions are met

    print(f'trying to reset. State: x={x}, theta={theta}, tripped={tripped_flag}, v={v}, thetadot={thetadot}')
    return


def load_checkpoints():
    """Load model checkpoints if a complete set exists.

    Only loads when the full set of checkpoint files (models + meta) is present
    to avoid mixing partial state. Restores `episode_count` and
    `learning_step` from the meta file if available.
    """
    global episode_count, learning_step, episode_rewards, wandb_run_id, abs_step_count
    global rl_step_count
    paths = _checkpoint_paths()
    required = [paths['critic'], paths['target_critic'], paths['actor'], paths['log_alpha'], paths['meta']]

    if not os.path.isdir(CHECKPOINT_DIR):
        print('No checkpoint directory found, starting from scratch')
        return

    if not all(os.path.exists(p) for p in required):
        print('Incomplete checkpoint set in', CHECKPOINT_DIR, '- skipping load')
        return

    try:
        critic.load_state_dict(torch.load(paths['critic'], map_location=device))
        target_critic.load_state_dict(torch.load(paths['target_critic'], map_location=device))
        actor.load_state_dict(torch.load(paths['actor'], map_location=device))
        v = torch.load(paths['log_alpha'], map_location=device)
        try:
            with torch.no_grad():
                log_alpha.copy_(v)
        except Exception:
            try:
                log_alpha.data.copy_(v)
            except Exception:
                pass
        print('Loaded checkpoints from', CHECKPOINT_DIR)
    except Exception as e:
        print('Failed to load checkpoints:', e)
        return

    # load meta info (episode_count, learning_step, episode_rewards, wandb_run_id, abs_step_count)
    try:
        meta = torch.load(paths['meta'], map_location=device)
        episode_count = int(meta.get('episode_count', episode_count))
        learning_step = int(meta.get('learning_step', learning_step))
        abs_step_count = int(meta.get('abs_step_count', abs_step_count))
        # wandb refuses a step lower than one already written, so a resumed run must
        # continue the RL axis where it left off, not restart it at 0
        rl_step_count = int(meta.get('rl_step_count', rl_step_count))
        episode_rewards = meta.get('episode_rewards', [])
        wandb_run_id = meta.get('wandb_run_id', None)
        print(f"Restored meta: episode_count={episode_count}, learning_step={learning_step}, abs_step_count={abs_step_count}, episodes_logged={len(episode_rewards)}, wandb_run_id={wandb_run_id}")
    except Exception:
        pass


def save_checkpoints():
    """Save model checkpoints to disk."""
    paths = _checkpoint_paths()
    try:
        # ensure directory exists only when saving
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(critic.state_dict(), paths['critic'])
        torch.save(target_critic.state_dict(), paths['target_critic'])
        torch.save(actor.state_dict(), paths['actor'])
        torch.save(log_alpha.detach().cpu(), paths['log_alpha'])
        # save meta info (episode_count, learning_step, episode_rewards, wandb_run_id, abs_step_count)
        meta = {
            'episode_count': episode_count, 
            'learning_step': learning_step,
            'abs_step_count': abs_step_count,
            'rl_step_count': rl_step_count,
            'episode_rewards': episode_rewards,
            'wandb_run_id': wandb_run_id
        }
        torch.save(meta, paths['meta'])
        print('Saved checkpoints to', CHECKPOINT_DIR)
    except Exception as e:
        print('Failed to save checkpoints:', e)


def _checkpoint_paths():
    return {
        'critic': os.path.join(CHECKPOINT_DIR, 'critic.pth'),
        'target_critic': os.path.join(CHECKPOINT_DIR, 'target_critic.pth'),
        'actor': os.path.join(CHECKPOINT_DIR, 'actor.pth'),
        'log_alpha': os.path.join(CHECKPOINT_DIR, 'log_alpha.pth'),
        'meta': os.path.join(CHECKPOINT_DIR, 'meta.pth'),
    }


# ---- final evaluation episode -------------------------------------------------
# Steps of the closing eval episode. Kept at max_ep_steps so the episodic return is
# directly comparable to the training episodes; raise it to watch the policy hold
# the pole for longer than it ever had to during training.
EVAL_MAX_STEPS = max_ep_steps

# How often the actor is queried during eval. Mirrors the training loop's N so the
# evaluated controller is the one that was trained, not a different k-step variant.
EVAL_N = N_STEP

# Number of repeated closing eval episodes (same frozen policy, independent physical
# trials on hardware) run on exit.
NUM_EVAL_RUNS = 20


def _fmt_param(param_t):
    """Format a parameter tensor as one CSV field (';'-joined if multi-dimensional)."""
    values = param_t.detach().cpu().numpy().reshape(-1)
    return values[0] if values.size == 1 else ';'.join(f'{v:.6g}' for v in values)


def run_eval_episode(
    max_steps: int = EVAL_MAX_STEPS,
    n_cycle: int = EVAL_N,
    eval_run: int = 1,
    seed: int = 0,
):
    """Run one final, noise-free episode and dump the full trajectory to a CSV.

    Deliberately different from a training episode:
      - the actor is queried with `deterministic=True`, so the squashed Gaussian
        returns its mode instead of a sample: this measures the learned policy,
        not the exploration policy wrapped around it
      - nothing is written to the replay buffer and no learner token is released,
        so the networks stay frozen for the whole episode
      - nothing goes to wandb: the run's step axis has already been closed out by
        the training loop, and back-filling it would corrupt the curves

    The CSV is written incrementally (so an abort still leaves usable data) and the
    summary is appended as '#'-prefixed trailer lines, which `pandas.read_csv(...,
    comment='#')` skips.

    Args:
        eval_run: Index of this eval run (1-based), used in the CSV filename to
            distinguish repeated trials of the same frozen policy.
        seed: Training seed, used in the CSV filename to identify which trained
            run this eval trial belongs to.

    Returns:
        The path of the written CSV, or None if the episode could not be started.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    csv_path = os.path.join(
        CHECKPOINT_DIR, f"eval_{PLANNER_NAME}_run{eval_run}_seed{seed}.csv"
    )

    print("\n" + "=" * 60)
    print(f"FINAL EVAL EPISODE (deterministic, no wandb) -> {csv_path}")
    print("Reset the pole to hanging; the run starts once it has settled.")
    print("=" * 60)

    # bring the cart back to a defined start state, then re-arm the Arduino
    reset_env()
    talk_to_arduino(0, mode=0)

    # take the freshest state the listener has
    with state_que.mutex:
        state_que.queue.clear()
    state = state_que.get()
    x, theta, v, thetadot, tripped = state
    thetadot = np.clip(thetadot, -20.0, 20.0)
    state = (x, theta, v, thetadot, tripped)

    # warm up the solver on the real state, exactly like a training episode does
    obs_init_batch = sac_state_to_tensor(state, batch=False).to(device).unsqueeze(0)
    with torch.no_grad():
        ctx_planner, _, _, _, _ = planner(obs_init_batch, ctx=None)
        pi_out_init = actor(obs_init_batch, ctx_planner, deterministic=True)
        ctx_current = pi_out_init.ctx

    step = 0
    cum_reward = 0.0
    step_in_cycle = 0
    param_current = None
    max_force = 0.0
    upright_streak = 0
    longest_upright = 0
    stabilized = False
    stabilized_at = None
    eval_theta_buffer = deque(maxlen=STABILIZATION_BUFFER_SIZE)
    solve_times, successes = [], []
    terminated = False

    columns = [
        'step', 'time_s', 'x_counts', 'x_m', 'theta_rad', 'v_counts_s', 'v_m_s',
        'thetadot_rad_s', 'tripped', 'param', 'u_force_N', 'u_pwm', 'reward',
        'cum_reward', 'upright', 'actor_called', 'solve_success', 'solve_retry',
        'solve_time_ms',
    ]

    done = done_eval(state, step, max_steps, x_threshold=_x_thr)
    t0 = time.perf_counter()
    f = open(csv_path, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(columns)

    try:
        while not done:
            step += 1
            obs_batch = sac_state_to_tensor(state, batch=False).to(device).unsqueeze(0)

            if step_in_cycle == 0:
                with torch.no_grad():
                    # deterministic=True -> mode of the distribution, no action noise
                    pi_out = actor(obs_batch, ctx_current, deterministic=True)
                param_current = pi_out.param[0].detach()
                ctx_current = pi_out.ctx
                u_force = float(pi_out.action[0].cpu().numpy().squeeze())
                actor_called = True
                stats = pi_out.stats if getattr(pi_out, 'stats', None) else {}
            else:
                with torch.no_grad():
                    # ctx passed by keyword: the positional form would land in the
                    # planner's `action` slot and silently drop the warm start
                    ctx_current, action, _, _, _ = planner(
                        obs_batch, param=param_current.unsqueeze(0), ctx=ctx_current
                    )
                u_force = float(action[0].cpu().numpy().squeeze())
                actor_called = False
                stats = ctx_current.log if getattr(ctx_current, 'log', None) else {}

            u_pwm = force_to_pwm(
                u_force, countpersecond_to_meterspersecond(v), max_pwm_limit=150
            )
            max_force = max(max_force, abs(u_force))
            talk_to_arduino(u_pwm, mode=0)

            # pre-step values, paired below with the reward they produced
            row_pre = [
                step, time.perf_counter() - t0, x, counts_to_meters(x), theta,
                v, countpersecond_to_meterspersecond(v), thetadot, int(tripped),
                _fmt_param(param_current), u_force, u_pwm,
            ]

            # wait for the next frame and drain to the freshest one
            state = state_que.get()
            while True:
                try:
                    state = state_que.get_nowait()
                except queue.Empty:
                    break

            x, theta, v, thetadot, tripped = state
            thetadot = np.clip(thetadot, -20.0, 20.0)
            state = (x, theta, v, thetadot, tripped)

            reward = compute_reward(state, u_force)
            cum_reward += reward

            theta_normalized = ((theta + np.pi) % (2 * np.pi)) - np.pi
            eval_theta_buffer.append(theta_normalized)
            is_upright = abs(theta_normalized) <= STABILIZATION_THRESHOLD
            if is_upright:
                upright_streak += 1
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
                reward, cum_reward, int(is_upright), int(actor_called),
                success, retry, solve_ms,
            ])
            f.flush()  # crash/abort safe

            done = done_eval(state, step, max_steps, x_threshold=_x_thr)
            terminated = bool(tripped) or abs(counts_to_meters(x)) > float(_x_thr)

            step_in_cycle += 1
            if step_in_cycle == n_cycle:
                step_in_cycle = 0

    except KeyboardInterrupt:
        print("\nEval episode aborted by user - partial trajectory kept.")
    finally:
        try:
            talk_to_arduino(0, mode=1)
        except Exception:
            pass
        trailer = {
            'planner': PLANNER_NAME,
            'eval_run': eval_run,
            'seed': seed,
            'deterministic': True,
            'steps': step,
            'max_steps': max_steps,
            'n_cycle': n_cycle,
            'episodic_return': round(cum_reward, 4),
            'mean_reward_per_step': round(cum_reward / step, 4) if step else float('nan'),
            'terminated_by_trip': int(terminated),
            'stabilized': int(stabilized),
            'stabilized_at_step': stabilized_at,
            'longest_upright_steps': longest_upright,
            'upright_fraction': round(longest_upright / step, 4) if step else float('nan'),
            'max_force_N': round(max_force, 4),
            'mean_solve_time_ms': round(float(np.mean(solve_times)), 4) if solve_times else float('nan'),
            'solver_success_rate': round(float(np.mean(successes)), 4) if successes else float('nan'),
            'episode_count_at_eval': episode_count,
            'abs_step_count_at_eval': abs_step_count,
        }
        for key, value in trailer.items():
            f.write(f"# {key}: {value}\n")
        f.close()

    print("-" * 60)
    for key, value in trailer.items():
        print(f"  {key}: {value}")
    print(f"Trajectory written to {csv_path}")
    print("-" * 60)
    return csv_path


def main():
    # seeding
    seed = 2
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    
    
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # load checkpoints/trained models if available
    # this means training progress persists across restarts
    # it consists of actor, critic, target_critic, log_alpha + meta info. Not the buffer
    load_checkpoints()

    # initialize wandb (resume if we have a run_id, else create new)
    global wandb_run_id

    # shared config for both branches; the planner/* keys record which OCP
    # formulation produced the run, so variants stay comparable after the fact
    wandb_config = {
        "buffer_size": cfg_saczop.buffer_size,
        "batch_size": cfg_saczop.batch_size,
        "lr_q": cfg_saczop.lr_q,
        "lr_pi": cfg_saczop.lr_pi,
        "lr_alpha": cfg_saczop.lr_alpha,
        "gamma": cfg_saczop.gamma,
        "tau": cfg_saczop.tau,
        "max_ep_steps": max_ep_steps,
        **planner_config_dict(PLANNER_NAME, cfg_planner),
    }

    if wandb_run_id:
        print(f"Resuming wandb run: {wandb_run_id}")
        wandb.init(
            project="cartpole-sac-zop-1",
            id=wandb_run_id,
            resume="allow",
            config=wandb_config,
        )
    else:
        print("Starting new wandb run")
        run = wandb.init(
            project="Paper-Real_SAC-ZOP",
            name=f"{PLANNER_NAME}_ocp_s{int(seed)}",
            config=wandb_config,
        )
        wandb_run_id = run.id
        print(f"New wandb run ID: {wandb_run_id}")
    
    # ensures we reference the module-level variables
    global ctx, episode_step_count, episode_count, abs_step_count, rl_step_count, learning_step, current_episode_reward, episode_rewards, max_force_perep, theta_buffer, stabilized_this_episode, num_terminations, num_stabilized_steps, upright_streak, longest_upright_steps, stabilized_at_step

    # env-throughput timing (for episode/steps_per_second, like the sim scripts)
    t_start = time.perf_counter()

    # start the background receiver task
    arduinoThread = threading.Thread(target=listen_to_arduino, args=())
    arduinoThread.daemon = True
    arduinoThread.start()

    # start the background learning task (use proper args)
    learningThread = threading.Thread(
        target=sac_zop_update_step,
        args=(cfg_saczop.batch_size, cfg_saczop.train_start),
    )
    learningThread.daemon = True
    learningThread.start()
    
    # save initial checkpoints at start of training
    print("Saving initial checkpoints...")
    save_checkpoints()
    
    
    #Main RL Loop#################################################################
    try:
        while True:
            # restart episode. Wait until env is reseted
            reset_env()
            
            # env is reseted so set new mode
            talk_to_arduino(0, mode=0)  # -> arduino is ready for normal operation
            
            # wait for the learner thread to catch up on all queued transitions
            print("Training inbetween episodes...")
            inbetween_training()
            
            # reset ctx
            ctx = None
            
            # reset step count
            episode_step_count = 0
            
            # reset episode reward
            current_episode_reward = 0.0

            # episodic return on the RL (macro-step) axis: sum of the per-cycle MEAN
            # rewards. Directly comparable to the 50 ms / N=1 runs' cumulative_reward,
            # whereas current_episode_reward now sums N x more env steps and is ~N x larger.
            current_episode_rl_return = 0.0
            
            # reset max force tracker
            max_force_perep = 0  # Fixed variable name
            
            # reset stabilization tracking
            theta_buffer.clear()
            stabilized_this_episode = False
            stabilized_at_step = None
            upright_streak = 0
            longest_upright_steps = 0

            # per-episode bookkeeping
            episode_count += 1
            
            # save checkpoints at episode 0 and then every 50 episodes
            if episode_count == 0 or episode_count % 50 == 0:
                print(f"Saving checkpoints at episode {episode_count}...")
                save_checkpoints()
                
                # generate and save policy heatmaps
                print(f"Generating policy heatmaps at episode {episode_count}...")
                try:
                    actor_path = os.path.join(CHECKPOINT_DIR, 'actor.pth')
                    # plot_policy_heatmap now saves PDFs automatically to checkpoint dir
                    plot_policy_heatmap(
                        actor_path=actor_path,
                        v_fixed=0.0,
                        thetadot_fixed=0.0,
                        x_range=(-0.35, 0.35),
                        theta_range=(-np.pi, np.pi),
                        resolution=50,
                        plt_show=False,
                        save_path=None,  # not used anymore, PDFs saved to checkpoint dir
                        episode_num=episode_count  # add episode number for filename
                    )
                    print(f"Saved policy heatmap PDFs to {CHECKPOINT_DIR}")
                    
                    # log PDFs to wandb as artifacts
                    try:
                        artifact = wandb.Artifact(
                            name=f'policy_heatmaps_ep{episode_count}',
                            type='plots',
                            description=f'Policy heatmap visualizations at episode {episode_count}'
                        )
                        
                        pdf_files = [
                            f'policy_heatmap_theta_ref_grid_ep{episode_count}.pdf',
                            f'policy_heatmap_force_grid_ep{episode_count}.pdf',
                            f'policy_heatmap_critic_grid_ep{episode_count}.pdf',
                            f'policy_heatmap_mpc_force_grid_ep{episode_count}.pdf',
                        ]
                        
                        # add each PDF file to the artifact
                        for pdf_name in pdf_files:
                            pdf_path = os.path.join(CHECKPOINT_DIR, pdf_name)
                            if os.path.exists(pdf_path):
                                artifact.add_file(pdf_path, name=pdf_name)
                        
                        # log the artifact
                        wandb.log_artifact(artifact)
                        print(f"Logged policy heatmap PDFs to wandb as artifact")
                    except Exception as e:
                        print(f"Failed to log PDFs to wandb: {e}")
                except Exception as e:
                    print(f"Failed to generate policy heatmaps: {e}")

            # reset state_que
            state_que.queue.clear()
            
            # wait for first valid state
            state = None
            state = state_que.get() # blocks until the thread adds a first state
            x, theta, v, thetadot, tripped = state 
            
            # clip thetadot to reasonable bounds before initialization
            thetadot = np.clip(thetadot, -20.0, 20.0)
            state = (x, theta, v, thetadot, tripped)
            
            # initialize MPC solver with the first real state instead of fixed x0
            obs_init = sac_state_to_tensor(state, batch=False).to(device)
            obs_init_batch = obs_init.unsqueeze(0)
            
            # first initialize the planner/controller with the real state
            state_converted_init = obs_init_batch
            with torch.no_grad():
                ctx_planner, _, _, _, _ = planner(state_converted_init, ctx=None)
            
            # then initialize the actor with the planner context
            with torch.no_grad():
                pi_out_init = actor(obs_init_batch, ctx_planner, deterministic=False)
                ctx = pi_out_init.ctx  # save the initialized context
            
            print(f"Initialized MPC solver with real state: x={counts_to_meters(x):.3f}m, theta={theta:.3f}rad")
            
            # check if for some reason ep is alredy done
            done = done_eval(state, episode_step_count, max_ep_steps, x_threshold=_x_thr)
            
            # variables for N-step buffering
            N = N_STEP  # call actor every N steps (see N_STEP at the top)
            step_in_cycle = 0  # tracks position within N-step cycle
            episode_rl_step = 0  # completed cycles in this episode = RL steps
            cycle_stats = {}     # wandb row for the cycle in progress
            cycle_upright = 0
            cycle_pi_stats = {}
            accumulated_reward = 0.0  # accumulates reward over N steps
            obs_start_cycle = None  # observation at start of N-step cycle
            param_current = None  # current parameter to use for MPC
            ctx_current = ctx  # current context for MPC
            
            # episode loop
            print("================================")
            print("Starting new episode")
            print(f"Episode {episode_count}, Learning Step: {learning_step}, Total Steps: {abs_step_count}")

            while not done:            
                # step counting
                episode_step_count += 1
                abs_step_count += 1
                
                # make state ready for actor/planner
                obs = sac_state_to_tensor(state, batch=False).to(device)
                obs_batch = obs.unsqueeze(0)
                
                # decide whether to call actor or just use existing params
                if step_in_cycle == 0:
                    # beginning of N-step cycle: call actor to get new params
                    obs_start_cycle = obs.clone()  # save for buffer
                    accumulated_reward = 0.0  # reset accumulator
                    
                    with torch.no_grad():
                        pi_out = actor(obs_batch, ctx_current, deterministic=False)
                    
                    # extract and save param for next N steps
                    param_current = pi_out.param[0].detach()
                    ctx_current = pi_out.ctx  # update context
                    
                    # get action (force) from pi_out
                    u_force = float(pi_out.action[0].cpu().numpy().squeeze())
                    actor_called = True
                    pi_stats = pi_out.stats if (hasattr(pi_out, 'stats') and pi_out.stats) else None

                else:
                    # intermediate step: re-solve the MPC on the CURRENT state with
                    # the param held from the cycle start.
                    with torch.no_grad():
                        # param/ctx passed by keyword: planner.forward is
                        # (obs, action=None, param=None, ctx=None), so the positional
                        # form would put ctx in the `action` slot and silently drop
                        # the warm start (same fix as in run_eval_episode)
                        ctx_current, action, _, _, _ = planner(
                            obs_batch, param=param_current.unsqueeze(0), ctx=ctx_current
                        )

                    # get force from planner output
                    u_force = float(action[0].cpu().numpy().squeeze())
                    actor_called = False
                    # solver stats live on the planner ctx on non-actor steps; without
                    # this, N-1 of every N steps would log no solver diagnostics
                    pi_stats = ctx_current.log if getattr(ctx_current, 'log', None) else None

                # convert force to PWM
                u_pwm = force_to_pwm(u_force, countpersecond_to_meterspersecond(v), max_pwm_limit=150)

                # track maximum absolute force
                max_force_perep = max(max_force_perep, abs(u_force))

                # actuate first, then log: keeps wandb work out of the
                # observe -> act latency path
                talk_to_arduino(u_pwm, mode=0)

                # Open a new wandb row at the cycle start. The state/action captured
                # here is what the ACTOR conditioned on and chose - at N=1 exactly
                # the value the old per-env-step logging recorded.
                if step_in_cycle == 0:
                    cycle_stats = {
                        'step/u_force_s': u_force,
                        'step/u_pwm_s': u_pwm,
                        'step/param_s': param_current.cpu().numpy().tolist() if param_current.dim() > 0 else param_current.item(),
                        'step/x_s': x,
                        'step/theta_s': theta,
                        'step/v_s': v,
                        'step/thetadot_s': thetadot,
                    }
                    cycle_upright = 0    # env steps spent upright in this cycle
                    cycle_pi_stats = {}  # solver stat name -> values over the cycle

                # Solver stats from every env step of the cycle, actor step or not,
                # collected under their original names so they land in the same
                # step/pi_*_s metrics as before - averaged instead of one-per-step.
                if pi_stats:
                    for key, val in pi_stats.items():
                        try:
                            cycle_pi_stats.setdefault(key, []).append(float(val))
                        except (TypeError, ValueError):
                            pass

                # wait for next state and drain queue to freshest
                state = state_que.get()
                while True:
                    try: 
                        state = state_que.get_nowait()
                    except queue.Empty:
                        break
                
                # clip thetadot
                x, theta, v, thetadot, tripped = state
                thetadot = np.clip(thetadot, -20.0, 20.0)
                state = (x, theta, v, thetadot, tripped)
                
                # make next state ready
                obs_next = sac_state_to_tensor(state, batch=False).to(device)
                
                # compute reward for this step
                reward = compute_reward(state, u_force)
                
                # accumulate reward over N-step cycle
                accumulated_reward += reward
                
                # accumulate episode reward
                current_episode_reward += reward
                
                # track stabilization: add theta to buffer and check if stabilized
                # normalize theta to [-pi, pi] range around 0 (upright position)
                theta_normalized = ((theta + np.pi) % (2 * np.pi)) - np.pi
                theta_buffer.append(theta_normalized)
                
                theta_abs = abs(theta_normalized)
                is_upright = theta_abs <= STABILIZATION_THRESHOLD

                # longest consecutive upright run: the ungated version of the latch below
                if is_upright:
                    upright_streak += 1
                    longest_upright_steps = max(longest_upright_steps, upright_streak)
                    cycle_upright += 1
                else:
                    upright_streak = 0

                # check if pole is stabilized (all recent theta values within threshold)
                if not stabilized_this_episode and len(theta_buffer) == STABILIZATION_BUFFER_SIZE:
                    if all(abs(t) <= STABILIZATION_THRESHOLD for t in theta_buffer):
                        stabilized_this_episode = True
                        # kept for the episode log below, so it lands on the episode axis
                        stabilized_at_step = episode_step_count

                # check done
                done = done_eval(state, episode_step_count, max_ep_steps, x_threshold=_x_thr)
                
                # increment cycle counter
                step_in_cycle += 1

                # terminated = ended by a trip (safety flag or |x| > X_TERM_M). Reaching
                # max_ep_steps is a truncation, not a termination: the value function must
                # still bootstrap there, so this - not `done` - is the buffer's terminal
                # flag (same as sim_sac_zop.py, which stores int(terminated)).
                terminated = bool(tripped) or abs(counts_to_meters(x)) > float(_x_thr)

                # store transition in buffer only at end of N-step cycle or if episode ends
                if step_in_cycle == N or done:
                    # MEAN over the cycle, not the sum: this is the reward RATE over
                    # the macro-step, which is exactly what a single 50 ms sample
                    # estimated in the old N=1 runs - so Q magnitudes, lr_q and the
                    # alpha/Q balance stay in their tuned range and old return curves
                    # remain on the same axis. Scaling by 1/N is a constant factor, so
                    # the optimal policy is identical to the summed version.
                    # Deliberately NOT "reward of the last step only": with thetadot up
                    # to 20 rad/s the pole turns up to 1 rad per macro-step, so a single
                    # boundary sample aliases the narrow `balanced` bonus away entirely
                    # (see compute_reward_cos_bonus_spin). Averaging all N samples has
                    # the same scale but ~N x lower variance and no aliasing.
                    # Divide by step_in_cycle, not N: a mid-cycle `done` ends the cycle
                    # early, and those cycles hold fewer than N accumulated rewards.
                    macro_reward = accumulated_reward / step_in_cycle
                    current_episode_rl_return += macro_reward

                    # this cycle IS one RL step: advance the wandb step axis
                    rl_step_count += 1
                    episode_rl_step += 1

                    # close the wandb row opened at the cycle start
                    try:
                        cycle_stats['step/reward_s'] = macro_reward
                        cycle_stats['step/stabilized_s'] = int(stabilized_this_episode)
                        # fraction of the cycle spent upright. At N=1 this is the same
                        # raw 0/1 signal as before, so averaging it in the wandb UI
                        # still gives the "fraction of time upright" curve.
                        cycle_stats['step/upright_s'] = cycle_upright / step_in_cycle
                        cycle_stats['step/episode_step_s'] = episode_rl_step
                        cycle_stats['step/abs_step_s'] = rl_step_count
                        # env steps folded into this row (N, or fewer on a cycle cut
                        # short by `done`)
                        cycle_stats['step/cycle_step_s'] = step_in_cycle
                        # the row always represents the actor step of the cycle
                        cycle_stats['step/actor_called_s'] = 1
                        for key, vals in cycle_pi_stats.items():
                            cycle_stats[f'step/pi_{key}_s'] = _safe_mean(vals)
                        wandb.log(cycle_stats, step=rl_step_count)
                    except Exception:
                        pass

                    # store accumulated N-step transition
                    param_t = param_current.to(replay_buffer.device).float()
                    replay_buffer.put((obs_start_cycle, param_t, float(macro_reward), obs_next, int(terminated)))
                    
                    # notify background learner (counted for drain bookkeeping)
                    try:
                        _note_token_released()
                    except Exception:
                        pass
                    
                    # reset cycle counter
                    step_in_cycle = 0
                
                # handle episode termination
                if done:
                    talk_to_arduino(0, mode=1)
                    if terminated:
                        num_terminations += 1
                    # every step of a stabilized episode counts as stabilized (not just
                    # the ones after the latch fired). wandb steps that are already
                    # written cannot be revised, so the credit is applied here in one go.
                    # credited in RL steps, so the total stays on the same scale as
                    # the 50 ms / N=1 runs
                    if stabilized_this_episode:
                        num_stabilized_steps += episode_rl_step
                    episode_rewards.append({
                        'episode': episode_count,
                        'cumulative_reward': current_episode_rl_return,
                        'steps': episode_rl_step
                    })
                    print(f"Episode {episode_count} finished: Total Reward = {current_episode_reward:.2f}, Steps = {episode_step_count}, Max Force = {max_force_perep}, Stabilized = {stabilized_this_episode}")

                    try:
                        ep_log = {
                            # episode counter: pick this as the panel x-axis in wandb to
                            # plot any *_e metric over episodes instead of env steps
                            'episode/episode_number_e': episode_count,
                            # episodic return on the RL axis: the sum of the per-cycle
                            # MEAN rewards, which is what a 50 ms / N=1 run measured by
                            # summing one reward per step. Directly overlayable on those.
                            'episode/cumulative_reward_e': current_episode_rl_return,
                            # RL steps, i.e. actor decisions - 200 for a 10 s episode at
                            # either control rate
                            'episode/steps_e': episode_rl_step,
                            'episode/max_force_e': max_force_perep,
                            # 1 if the pole was balanced for STABILIZATION_BUFFER_SIZE
                            # consecutive steps anywhere in this episode, else 0
                            'episode/stabilized_e': int(stabilized_this_episode),
                            'terminations/total_e': num_terminations,
                            'episode/learning_step_e': learning_step,
                            # graded version of `stabilized` - see longest_upright_steps.
                            # Converted from env steps to RL steps so the number means the
                            # same thing it did at 50 ms / N=1.
                            'episode/longest_upright_steps_e': longest_upright_steps / N,
                            # all RL steps of a stabilized episode, 0 otherwise
                            'episode/stabilized_steps_e': episode_rl_step if stabilized_this_episode else 0,
                            'perf/stabilized_steps_total_e': num_stabilized_steps,
                        }
                        # only defined for episodes that actually stabilized. Converted
                        # to RL steps for the same reason as longest_upright_steps_e.
                        if stabilized_at_step is not None:
                            ep_log['stabilization/achieved_at_step_e'] = stabilized_at_step / N
                        wandb.log(ep_log, step=rl_step_count)
                    except Exception:
                        pass

    except KeyboardInterrupt:
        # try to stop actuator and plot
        try:
            talk_to_arduino(0, mode=1)  # tell arduino to stop
            print(f'serial closed')
        except Exception:
            pass
        # print training step timing statistics
        if training_step_times:
            avg_time = np.mean(training_step_times)
            max_time = np.max(training_step_times)
            print(f"\nTraining step timing statistics:")
            print(f"  Total training steps: {len(training_step_times)}")
            print(f"  Average time per step: {avg_time*1000:.2f} ms")
            print(f"  Maximum time per step: {max_time*1000:.2f} ms")
        # save model checkpoints on exit so training progress persists
        try:
            save_checkpoints()
        except Exception:
            pass

        # closing evaluation of the final policy: deterministic, unlogged, dumped
        # to CSV, repeated NUM_EVAL_RUNS times as independent physical trials of the
        # same frozen policy. Runs after save_checkpoints() so the weights on disk
        # are exactly the ones being evaluated. A further Ctrl+C aborts all
        # remaining eval runs.
        if RUN_EVAL_ON_EXIT:
            try:
                for eval_run in range(1, NUM_EVAL_RUNS + 1):
                    print(f"\n>>> Eval run {eval_run}/{NUM_EVAL_RUNS} (seed={seed})")
                    run_eval_episode(eval_run=eval_run, seed=seed)
            except KeyboardInterrupt:
                print("Remaining eval runs skipped.")
            except Exception as e:
                print(f"Final eval episode failed: {e}")

        # finish wandb run
        # try:
        #     wandb.finish()
        # except Exception:
        #     pass
    finally:
        ser.close()
        pass



if __name__ == "__main__":
    main()

