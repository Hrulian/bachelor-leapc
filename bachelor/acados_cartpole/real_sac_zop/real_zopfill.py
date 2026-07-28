"""Two-phase MPC-fill -> pure-SAC experiment ON THE REAL HARDWARE.

This is the hardware twin of ``simulation_zaczop/sim_sac_zopfill.py``: it runs the
exact same two-phase idea, but with the real serial/Arduino control logic from
``real_sac_zop.py`` (background listener thread, background learner thread,
N-step MPC cycling, reset handshake, force->PWM, ...).

Phases
------
    Phase 1  (steps 0 .. SWITCH_STEP):   an UNTRAINED SAC-ZOP actor (MPC layer +
             Gaussian noise in the MPC parameter space around the defaults)
             DRIVES the real cart and fills the replay buffer. It is never
             trained. Meanwhile the pure SAC agent is warmed up in the BACKGROUND
             learner thread off-policy on those transitions.

    Phase 2  (steps SWITCH_STEP .. end):  the pure SAC baseline (plain SacActor,
             force = 1-D action) takes over DRIVING and keeps learning normally.
             From here on it is completely ordinary SAC on a pre-filled buffer.

Buffer compatibility (important!)
---------------------------------
Native SAC-ZOP stores the MPC *parameter vector* as the buffer action; pure SAC
needs the scalar *force*. They are incompatible. So in BOTH phases we store the
**force actually applied to the cart** as a shape-(1,) tensor — exactly the
format the pure-SAC critic/actor consume. The buffer format is therefore
identical across both phases; only *who* generated the force differs.

The user asked for: drive 5k steps with an untrained SAC-ZOP policy, then let the
pure SAC agent take over (as in zopfill). Hence SWITCH_STEP = 5000 below.
"""

import torch, threading, queue, serial, time, os, csv
import numpy as np
import gymnasium as gym
import wandb
from collections import deque
from bachelor.acados_cartpole.real_sac_zop.my_sac import (
    SacActor,
    SacCritic,
    SacTrainerConfig,
)
from bachelor.acados_cartpole.real_sac_zop.my_sac_zop import MpcSacActor, SacZopTrainerConfig
from bachelor.acados_cartpole.real_sac_zop.my_utils import soft_target_update
from bachelor.acados_cartpole.real_sac_zop.my_buffer import ReplayBuffer
from bachelor.acados_cartpole.my_planner import (
    CartPolePlannerConfig,
    CartPolePlanner,
    create_custom_cartpole_params,
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

from leap_c.planner import ControllerFromPlanner
from leap_c.torch.nn.extractor import get_extractor_cls  # optional helper


# TWO-PHASE CONFIG #################################################################
SWITCH_STEP = 5000   # env step at which we switch Phase-1 (SAC-ZOP fill) -> Phase-2 (pure SAC)
FILL_TRAIN_START = 1000  # background pure-SAC training begins after this many collected steps


#COMMUNICATION#####################################################################
# wandb communication parameters
os.environ['WANDB_API_KEY'] = 'fd053eb0471b83f999819cd4c4e4930ea28de0ea'

# serial communication parameters
PORT = "/dev/ttyACM0"
BAUD = 115200
FRAME_TIMEOUT_S = 0.05   #if for 50 ms nothing arrived discard this frame
MAX_PAYLOAD_LEN = 64
READ_TIMEOUT_S = 0.002

ser = serial.Serial(PORT, BAUD, timeout=READ_TIMEOUT_S)
time.sleep(2)
ser.reset_input_buffer()

state_que = queue.Queue() # init que that stores states
prev_sent_mode = None  # remembers last mode sent to Arduino for logging
# checkpoint directory for models
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints_zopfill")

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


# initialize counters
episode_step_count = 0 # steps in current episode
total_training_steps = 0  # einzelne training steps (critic + actor updates) - für soft_update_freq
learning_step = 0   # blocks (20-step blocks) - für logging und tracking
episode_count = 0   # total episodes completed
abs_step_count = 0  # total steps across all episodes

# episode reward tracking
episode_rewards = []  # list of cumulative rewards per episode
current_episode_reward = 0.0  # accumulated reward in current episode
max_force_perep = 0  # track max force per episode
wandb_run_id = None  # wandb run ID for resuming runs

# metrics mirrored from the sim scripts (sim_sac / sim_sac_zop / sim_sac_zopfill)
num_terminations = 0  # cumulative episodes that ended in a trip (terminated, not truncated)
num_stabilized_steps = 0  # cumulative steps spent in the balanced/stabilized mode

# stabilization tracking
STABILIZATION_BUFFER_SIZE = 200  # ~1 second at 10ms sample time
STABILIZATION_THRESHOLD = 0.15  # rad, ±0.15 rad around upright
theta_buffer = deque(maxlen=STABILIZATION_BUFFER_SIZE)  # FIFO queue for theta values
stabilized_this_episode = False  # flag: was pole stabilized this episode

# max steps per episode
max_ep_steps = 200  # 10 s at the 50 ms control rate

# device setup
device = "cpu"
# MPC Layer Setup (Phase-1 filler only) ###########################################
cfg_planner = CartPolePlannerConfig()
params = create_custom_cartpole_params("global", cfg_planner.N_horizon)
planner = CartPolePlanner(cfg_planner, params)
controller_wrapped = ControllerFromPlanner(planner)
ctx = None


# observation space
# state is (x, theta, xdot, thetadot)
# use planner config for reasonable x-bounds and clamp angle to [-pi, pi]
# episode termination threshold - deliberately NOT cfg_planner.x_threshold: that value is
# the MPC's own hard box constraint on x (my_acados_ocp.py), a controller design choice.
# Termination is an environment property and is shared with real_sac.py via X_TERM_M.
_x_thr = X_TERM_M
_x_low = -float(_x_thr) # in meters
_x_high = float(_x_thr) # in meters

obs_low = np.array([_x_low, -np.pi, -5, -21], dtype=np.float32)
obs_high = np.array([_x_high, np.pi, 5, 21], dtype=np.float32)
obs_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

# Phase-1 filler action space = MPC parameter vector; Phase-2 (pure SAC) action
# space = scalar cart force (1-D), bounded by the planner's Fmax. The replay
# buffer and pure-SAC networks operate on the FORCE space.
param_space = controller_wrapped.param_space
Fmax = float(cfg_planner.Fmax)
action_space = gym.spaces.Box(low=-Fmax, high=Fmax, shape=(1,), dtype=np.float32)


# ---- Phase-2 pure-SAC networks (these are the ONLY networks that get trained) ---
cfg_sac = SacTrainerConfig()
cfg_sac.critic_mlp.norm_layer = "layer_norm"  # match sim_sac_zopfill.py

# Replay Buffer init (stores FORCE actions, shape (1,))
replay_buffer = ReplayBuffer(buffer_limit=cfg_sac.buffer_size, device=device)

extractor_cls = get_extractor_cls("identity")

critic = SacCritic(
    extractor_cls=extractor_cls,
    observation_space=obs_space,
    action_space=action_space,
    mlp_cfg=cfg_sac.critic_mlp,
    num_critics=cfg_sac.num_critics,
).to(device)

target_critic = SacCritic(
    extractor_cls=extractor_cls,
    observation_space=obs_space,
    action_space=action_space,
    mlp_cfg=cfg_sac.critic_mlp,
    num_critics=cfg_sac.num_critics,
).to(device)
target_critic.load_state_dict(critic.state_dict())

actor = SacActor(
    extractor_cls=extractor_cls,
    observation_space=obs_space,
    action_space=action_space,
    distribution_name=cfg_sac.distribution_name,
    mlp_cfg=cfg_sac.actor_mlp,
).to(device)

# entropy temperature alpha init
log_alpha = torch.nn.Parameter(
    torch.tensor(cfg_sac.init_alpha, dtype=torch.float32).log()
).to(device)

alpha_optimizer = (
    torch.optim.Adam([log_alpha], lr=cfg_sac.lr_alpha)
    if cfg_sac.lr_alpha is not None
    else None
)

action_dim = action_space.shape[0]  # = 1
target_entropy = -float(action_dim)  # standard SAC heuristic for a 1-D action

critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg_sac.lr_q)
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg_sac.lr_pi)


# ---- Phase-1 filler: UNTRAINED SAC-ZOP actor (MPC + param-space noise) ----------
# Built once, kept in eval mode, and NEVER trained — it only drives during Phase 1.
cfg_saczop = SacZopTrainerConfig()
filler_actor = MpcSacActor(
    extractor_cls=extractor_cls,
    observation_space=obs_space,
    controller=controller_wrapped,
    distribution_name=cfg_saczop.distribution_name,
    mlp_cfg=cfg_saczop.actor_mlp,
    init_param_with_default=cfg_saczop.init_param_with_default,
).to(device)
filler_actor.eval()


def sac_update_step(batch_size, train_start):
    """
    Background thread: one training block (20 gradient steps) per buffer drop.
    1 semaphore token = 1 block. If tokens pile up (training slower than sampling)
    the thread runs back-to-back without waiting, using all available time.

    Trains the PURE SAC networks (actor/critic) on force transitions — during
    Phase 1 this is a background warm-up on the SAC-ZOP filler's data, during
    Phase 2 it is the normal on-policy-ish SAC learning.
    """
    global abs_step_count, learning_step, total_training_steps
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
                    sac_single_step_update(
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
                        'learning/training_block_time_ms': block_elapsed_time * 1000,
                        'learning/block_step': learning_step,
                        'learning/total_training_steps': total_training_steps,
                        # backlog: tokens released but not yet processed (lock-free read)
                        'learning/pending_tokens': _tokens_released - _tokens_processed,
                        'learning/q_loss_avg': np.mean(metrics_accumulator['q_losses']),
                        'learning/pi_loss': metrics_accumulator['pi_losses'][0] if metrics_accumulator['pi_losses'] else float('nan'),
                        'learning/alpha': metrics_accumulator['alphas'][0] if metrics_accumulator['alphas'] else float('nan'),
                        'learning/q_avg': np.mean(metrics_accumulator['q_values']),
                        'learning/q_target_avg': np.mean(metrics_accumulator['q_targets']),
                        'learning/entropy': metrics_accumulator['entropies'][0] if metrics_accumulator['entropies'] else float('nan'),
                    }, step=abs_step_count)
                except Exception:
                    pass

        except Exception as e:
            print("Exception in sac_update_step:", e)


def sac_single_step_update(batch_size, update_actor=True, metrics_accumulator=None):
    """
    Pure SAC training step (identical maths to sim_sac_zopfill.py's single_update):
    one critic (+ optional actor + alpha) update on FORCE transitions.
    """

    global total_training_steps

    if len(replay_buffer) < batch_size:
        return False

    o, a, r, o_prime, te = replay_buffer.sample(batch_size)

    # cache alpha once — avoids repeated exp() + .item() syncs
    alpha = log_alpha.exp().item()

    # critic target (no grad)
    with torch.no_grad():
        a_pi_prime, log_p_prime, _ = actor(o_prime)
        q_target = target_critic(o_prime, a_pi_prime)
        q_target = torch.min(q_target, dim=1, keepdim=True).values
        q_target = q_target - alpha * log_p_prime * cfg_sac.entropy_reward_bonus
        target = r[:, None].to(device) + cfg_sac.gamma * (1 - te[:, None].to(device)) * q_target

    # actor forward + alpha update BEFORE critic — only on actor steps
    if update_actor:
        a_pi, log_p, _ = actor(o)
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

    if total_training_steps % cfg_sac.soft_update_freq == 0:
        soft_target_update(critic, target_critic, cfg_sac.tau)

    total_training_steps += 1

    if metrics_accumulator is not None:
        metrics_accumulator['q_losses'].append(q_loss.item())
        metrics_accumulator['q_values'].append(q.mean().item())
        metrics_accumulator['q_targets'].append(target.mean().item())
        if update_actor:
            metrics_accumulator['pi_losses'].append(pi_loss.item())
            metrics_accumulator['alphas'].append(alpha)
            metrics_accumulator['entropies'].append(-log_p.mean().item())

    return True


def inbetween_training(timeout_s: float | None = 30.0):
    """
    Block until the background learner thread has drained all tokens released so
    far, i.e. every transition queued up to this point has been trained on.
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
    """Load pure-SAC model checkpoints if a complete set exists."""
    global episode_count, learning_step, episode_rewards, wandb_run_id, abs_step_count
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

    # load meta info
    try:
        meta = torch.load(paths['meta'], map_location=device)
        episode_count = int(meta.get('episode_count', episode_count))
        learning_step = int(meta.get('learning_step', learning_step))
        abs_step_count = int(meta.get('abs_step_count', abs_step_count))
        episode_rewards = meta.get('episode_rewards', [])
        wandb_run_id = meta.get('wandb_run_id', None)
        print(f"Restored meta: episode_count={episode_count}, learning_step={learning_step}, abs_step_count={abs_step_count}, episodes_logged={len(episode_rewards)}, wandb_run_id={wandb_run_id}")
    except Exception:
        pass


def save_checkpoints():
    """Save pure-SAC model checkpoints to disk."""
    paths = _checkpoint_paths()
    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(critic.state_dict(), paths['critic'])
        torch.save(target_critic.state_dict(), paths['target_critic'])
        torch.save(actor.state_dict(), paths['actor'])
        torch.save(log_alpha.detach().cpu(), paths['log_alpha'])
        meta = {
            'episode_count': episode_count,
            'learning_step': learning_step,
            'abs_step_count': abs_step_count,
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


def store_transition(obs_start, u_force, reward, obs_next, terminated):
    """Store a single-step FORCE transition (shape (1,)) and notify the learner.

    This is the zopfill buffer format: the action is ALWAYS the applied force, so
    the pure-SAC critic/actor can consume Phase-1 (SAC-ZOP) transitions too.

    `terminated` must be the true termination flag (trip / |x| > X_TERM_M), NOT the
    episode-`done` flag: a max_ep_steps truncation still has to bootstrap.
    """
    action_t = torch.tensor([float(u_force)], dtype=torch.float32).to(replay_buffer.device)
    replay_buffer.put((obs_start, action_t, float(reward), obs_next, int(terminated)))
    try:
        _note_token_released()
    except Exception:
        pass


def main():
    # load pure-SAC checkpoints/trained models if available (resumable)
    load_checkpoints()

    # initialize wandb (resume if we have a run_id, else create new)
    global wandb_run_id
    wandb_config = {
        "algo": "real_buffer_fill_then_pure_sac",
        "switch_step": SWITCH_STEP,
        "fill_train_start": FILL_TRAIN_START,
        "buffer_size": cfg_sac.buffer_size,
        "batch_size": cfg_sac.batch_size,
        "lr_q": cfg_sac.lr_q,
        "lr_pi": cfg_sac.lr_pi,
        "lr_alpha": cfg_sac.lr_alpha,
        "gamma": cfg_sac.gamma,
        "tau": cfg_sac.tau,
        "Fmax": Fmax,
        "max_ep_steps": max_ep_steps,
    }
    if wandb_run_id:
        print(f"Resuming wandb run: {wandb_run_id}")
        wandb.init(project="cartpole-mpc-fill-real", id=wandb_run_id,
                   resume="allow", config=wandb_config)
    else:
        print("Starting new wandb run")
        run = wandb.init(project="cartpole-mpc-fill-real",
                         name=f"real_zopfill_run_{int(time.time())}", config=wandb_config)
        wandb_run_id = run.id
        print(f"New wandb run ID: {wandb_run_id}")

    # ensures we reference the module-level variables
    global ctx, episode_step_count, episode_count, abs_step_count, learning_step, current_episode_reward, episode_rewards, max_force_perep, theta_buffer, stabilized_this_episode, num_terminations, num_stabilized_steps

    # env-throughput timing (for episode/steps_per_second, like the sim scripts)
    t_start = time.perf_counter()

    # start the background receiver task
    arduinoThread = threading.Thread(target=listen_to_arduino, args=())
    arduinoThread.daemon = True
    arduinoThread.start()

    # start the background learning task (trains the PURE SAC networks)
    learningThread = threading.Thread(
        target=sac_update_step,
        args=(cfg_sac.batch_size, FILL_TRAIN_START),
    )
    learningThread.daemon = True
    learningThread.start()

    # save initial checkpoints at start of training
    print("Saving initial checkpoints...")
    save_checkpoints()

    print("=" * 60)
    print(f"Phase 1 (0..{SWITCH_STEP}): UNTRAINED SAC-ZOP drives + fills buffer, "
          f"pure SAC warms up in background (from step {FILL_TRAIN_START})")
    print(f"Phase 2 ({SWITCH_STEP}..): pure SAC drives + learns")
    print("=" * 60)

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

            # reset counters / trackers
            episode_step_count = 0
            current_episode_reward = 0.0
            max_force_perep = 0
            theta_buffer.clear()
            stabilized_this_episode = False

            # per-episode bookkeeping
            episode_count += 1

            # save checkpoints at episode 0 and then every 50 episodes
            if episode_count == 0 or episode_count % 50 == 0:
                print(f"Saving checkpoints at episode {episode_count}...")
                save_checkpoints()

            # reset state_que
            state_que.queue.clear()

            # wait for first valid state
            state = state_que.get() # blocks until the thread adds a first state
            x, theta, v, thetadot, tripped = state

            # clip thetadot to reasonable bounds before initialization
            thetadot = np.clip(thetadot, -20.0, 20.0)
            state = (x, theta, v, thetadot, tripped)

            # does this episode START in the Phase-1 (SAC-ZOP fill) regime?
            ep_in_fill_phase = abs_step_count < SWITCH_STEP

            # initialize MPC solver with the first real state — only needed while
            # the SAC-ZOP filler is (or may become) the driver this episode.
            ctx_current = None
            if ep_in_fill_phase:
                obs_init = sac_state_to_tensor(state, batch=False).to(device)
                obs_init_batch = obs_init.unsqueeze(0)
                with torch.no_grad():
                    ctx_planner, _, _, _, _ = planner(obs_init_batch, ctx=None)
                    pi_out_init = filler_actor(obs_init_batch, ctx_planner, deterministic=False)
                    ctx_current = pi_out_init.ctx
                print(f"Initialized MPC solver with real state: x={counts_to_meters(x):.3f}m, theta={theta:.3f}rad")

            # check if for some reason ep is alredy done
            done = done_eval(state, episode_step_count, max_ep_steps, x_threshold=_x_thr)

            # variables for Phase-1 N-step MPC cycling (as in real_sac_zop.py)
            N = 5  # in Phase 1: call SAC-ZOP actor every N steps, planner in between
            step_in_cycle = 0
            param_current = None

            # episode loop
            print("================================")
            print("Starting new episode")
            print(f"Episode {episode_count}, Learning Step: {learning_step}, Total Steps: {abs_step_count}, "
                  f"phase={'fill' if ep_in_fill_phase else 'sac'}")

            while not done:
                # step counting
                episode_step_count += 1
                abs_step_count += 1

                # phase decision for THIS step
                fill_phase = abs_step_count <= SWITCH_STEP

                # make state ready for actor/planner
                obs = sac_state_to_tensor(state, batch=False).to(device)
                obs_batch = obs.unsqueeze(0)

                if fill_phase:
                    # -------- Phase 1: UNTRAINED SAC-ZOP drives (N-step cycling) --------
                    if step_in_cycle == 0:
                        # beginning of N-step cycle: call the SAC-ZOP actor for a new param
                        with torch.no_grad():
                            pi_out = filler_actor(obs_batch, ctx_current, deterministic=False)
                        param_current = pi_out.param[0].detach()
                        ctx_current = pi_out.ctx
                        u_force = float(pi_out.action[0].cpu().numpy().squeeze())
                        actor_called = True
                    else:
                        # intermediate step: reuse saved param with the MPC planner
                        with torch.no_grad():
                            ctx_current, action, _, _, _ = planner(
                                obs_batch, ctx_current, param_current.unsqueeze(0)
                            )
                        u_force = float(action[0].cpu().numpy().squeeze())
                        actor_called = False

                    step_in_cycle = (step_in_cycle + 1) % N
                else:
                    # -------- Phase 2: pure SAC drives (per-step force, no MPC) --------
                    with torch.no_grad():
                        act, _, _ = actor(obs_batch, deterministic=False)
                    u_force = float(act[0].cpu().numpy().squeeze())
                    actor_called = True
                    step_in_cycle = 0  # keep cycle reset so a re-entry would start clean

                # clip to the applied-force range (matches the stored action)
                u_force = float(np.clip(u_force, -Fmax, Fmax))

                # log step stats
                try:
                    wandb.log({
                        'step/u_force': u_force,
                        'step/x': x,
                        'step/theta': theta,
                        'step/v': v,
                        'step/thetadot': thetadot,
                        'step/episode_step': episode_step_count,
                        'step/fill_phase': int(fill_phase),
                        'step/actor_called': int(actor_called),
                        'step/stabilized': int(stabilized_this_episode),
                        'stabilization/stabilized_steps_total': num_stabilized_steps,
                        'terminations/total': num_terminations,
                    }, step=abs_step_count)
                except Exception:
                    pass

                # convert force to PWM
                u_pwm = force_to_pwm(u_force, countpersecond_to_meterspersecond(v), max_pwm_limit=150)

                # track maximum absolute force
                max_force_perep = max(max_force_perep, abs(u_force))

                # send PWM control to arduino
                talk_to_arduino(u_pwm, mode=0)

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
                current_episode_reward += reward

                # track stabilization
                theta_normalized = ((theta + np.pi) % (2 * np.pi)) - np.pi
                theta_buffer.append(theta_normalized)
                if not stabilized_this_episode and len(theta_buffer) == STABILIZATION_BUFFER_SIZE:
                    if all(abs(t) <= STABILIZATION_THRESHOLD for t in theta_buffer):
                        stabilized_this_episode = True
                        num_stabilized_steps += STABILIZATION_BUFFER_SIZE  # retroactively credit the 200-step balanced window
                        try:
                            wandb.log({
                                'stabilization/achieved_at_step': episode_step_count,
                                'stabilization/achieved_at_abs_step': abs_step_count,
                            }, step=abs_step_count)
                        except Exception:
                            pass
                elif stabilized_this_episode:
                    num_stabilized_steps += 1  # latched: every step after achievement counts as stabilized

                try:
                    wandb.log({'step/reward': reward}, step=abs_step_count)
                except Exception:
                    pass

                # check done
                done = done_eval(state, episode_step_count, max_ep_steps, x_threshold=_x_thr)

                # terminated = ended by a trip (safety flag or |x| > X_TERM_M). Reaching
                # max_ep_steps is a truncation, not a termination: the value function must
                # still bootstrap there, so this - not `done` - is the buffer's terminal flag.
                terminated = bool(tripped) or abs(counts_to_meters(x)) > float(_x_thr)

                # store SINGLE-STEP FORCE transition (both phases) + notify learner
                store_transition(obs, u_force, reward, obs_next, terminated)

                # handle episode termination
                if done:
                    talk_to_arduino(0, mode=1)
                    if terminated:
                        num_terminations += 1
                    elapsed = time.perf_counter() - t_start
                    sps = abs_step_count / max(elapsed, 1e-9)
                    episode_rewards.append({
                        'episode': episode_count,
                        'cumulative_reward': current_episode_reward,
                        'steps': episode_step_count
                    })
                    print(f"Episode {episode_count} [{'fill' if ep_in_fill_phase else 'sac'}] finished: "
                          f"Total Reward = {current_episode_reward:.2f}, Steps = {episode_step_count}, "
                          f"Max Force = {max_force_perep}, Stabilized = {stabilized_this_episode}")

                    try:
                        wandb.log({
                            'episode/episode_number': episode_count,
                            'episode/cumulative_reward': current_episode_reward,
                            'episode/steps': episode_step_count,
                            'episode/max_force': max_force_perep,
                            'episode/stabilized': int(stabilized_this_episode),
                            'episode/steps_per_second': sps,
                            'episode/terminated': int(terminated),
                            'episode/fill_phase': int(ep_in_fill_phase),
                            'terminations/total': num_terminations,
                        }, step=abs_step_count)
                    except Exception:
                        pass

    except KeyboardInterrupt:
        # try to stop actuator and print timing stats
        try:
            talk_to_arduino(0, mode=1)  # tell arduino to stop
            print(f'serial closed')
        except Exception:
            pass
        if training_step_times:
            avg_time = np.mean(training_step_times)
            max_time = np.max(training_step_times)
            print(f"\nTraining step timing statistics:")
            print(f"  Total training blocks: {len(training_step_times)}")
            print(f"  Average time per block: {avg_time*1000:.2f} ms")
            print(f"  Maximum time per block: {max_time*1000:.2f} ms")
        try:
            save_checkpoints()
        except Exception:
            pass
    finally:
        ser.close()
        pass


if __name__ == "__main__":
    main()
