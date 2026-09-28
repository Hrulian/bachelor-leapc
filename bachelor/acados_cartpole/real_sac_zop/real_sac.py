import torch, threading, queue, serial, time, os, csv
import numpy as np
import gymnasium as gym
import wandb
from collections import deque
from bachelor.acados_cartpole.real_sac_zop.my_sac import SacCritic, SacActor, SacTrainerConfig
from bachelor.acados_cartpole.real_sac_zop.my_utils import soft_target_update
from bachelor.acados_cartpole.real_sac_zop.my_buffer import ReplayBuffer
from bachelor.acados_cartpole.my_helpers import (
    force_to_pwm,
    countpersecond_to_meterspersecond,
    counts_to_meters,
    compute_reward,
    done_eval,
    sac_state_to_tensor,
    X_TERM_M,
)
from bachelor.acados_cartpole.my_utils_plot import plot_sac_policy_heatmap
 
from leap_c.torch.nn.extractor import get_extractor_cls

#COMMUNICATION#####################################################################
# wandb communication parameters
os.environ['WANDB_API_KEY'] = 'fd053eb0471b83f999819cd4c4e4930ea28de0ea'

# serial communication parameters
PORT = "/dev/ttyACM0"
BAUD = 115200
FRAME_TIMEOUT_S = 0.05
MAX_PAYLOAD_LEN = 64
READ_TIMEOUT_S = 0.002

ser = serial.Serial(PORT, BAUD, timeout=READ_TIMEOUT_S)
time.sleep(2)                 
ser.reset_input_buffer()  

state_que = queue.Queue()
prev_sent_mode = None
# own subfolder: real_sac and real_sac_zop use identical checkpoint filenames, and the
# plain-SAC critic state dict is shape-compatible with the ZOP one (same 4-dim obs, same
# 1-dim action/param), so a shared directory would silently cross-load models and, via
# meta.pth, resume into the other script's wandb run
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "real_sac")

# timing tracking for training blocks (one entry per 20-step block)
training_step_times = []

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
            if currently_receiving and (time.monotonic() - t0 > FRAME_TIMEOUT_S):
                currently_receiving = False
                buffer.clear()
                t0 = None
            continue 
        
        c = b[0]
        
        if currently_receiving == False: 
            if c == ord('<'):
                currently_receiving = True
                buffer.clear()
                t0 = time.monotonic()
            continue
        
        if c == ord('>'): 
            try:
                payload = buffer.decode('ascii', errors='strict').strip()
                parts = payload.split(',')
                if len(parts) == 5:
                    x, theta, v, thetadot = map(float, parts[:4])
                    tripped = int(parts[4])
                    state_que.put((x, theta, v, thetadot, tripped))
                else:
                    pass
                        
            except Exception:
                pass
                    
            currently_receiving = False
            buffer.clear()
            t0 = None
            continue
            
        elif c in (ord('\n'), ord('\r')):  
            continue
        
        if len(buffer) < MAX_PAYLOAD_LEN:
            buffer.append(c)
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
    try:
        if prev_sent_mode is None or prev_sent_mode != mode:
            print(f"[HOST MODE] mode -> {mode}")
            prev_sent_mode = mode
    except Exception:
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
episode_step_count = 0  # steps in current episode
total_training_steps = 0  # individual gradient steps (critic + actor) - for soft_update_freq
learning_step = 0  # blocks (20-step blocks) - for logging and tracking
episode_count = 0  # total episodes completed
abs_step_count = 0  # total steps across all episodes

# episode reward tracking
episode_rewards = []
current_episode_reward = 0.0
max_force_perep = 0
wandb_run_id = None

# metrics mirrored from the sim scripts (sim_sac / sim_sac_zop / sim_sac_zopfill)
num_terminations = 0  # cumulative episodes that ended in a trip (terminated, not truncated)
# cumulative steps credited as stabilized: an episode in which the latch ever fired
# contributes ALL of its steps, not just the ones after the latch
num_stabilized_steps = 0

# stabilization tracking
STABILIZATION_BUFFER_SIZE = 40  # 2 s of balancing at the 50 ms control rate
STABILIZATION_THRESHOLD = 0.15
theta_buffer = deque(maxlen=STABILIZATION_BUFFER_SIZE)
stabilized_this_episode = False
stabilized_at_step = None  # episode step at which the latch fired, None if it never did

# longest run of consecutive steps within STABILIZATION_THRESHOLD in the current episode.
# Same quantity the latch thresholds, but reported ungated: rotation tops out at a few
# steps while real balancing reaches the full episode, so it separates "hopeless" from
# "almost there" where the binary flag stays stuck at 0.
upright_streak = 0
longest_upright_steps = 0

# deliberately no pre-averaged metrics here: the raw per-step signals (step/reward_s,
# step/theta_s, step/upright_s) are logged unsmoothed and averaged in the wandb UI instead.
#
# Naming convention for every logged metric:
#   *_s  -> one value per env step, meant to be plotted over the step axis
#   *_e  -> one value per episode, meant to be plotted over episode/episode_number_e
# Everything still goes to wandb at step=abs_step_count (wandb has a single step axis);
# the suffix says which x-axis the metric is meant to be read on. To get the episode
# axis, pick episode/episode_number_e as the panel's x-axis in the wandb UI.

# max steps per episode
max_ep_steps = 200  # 15 s at the 50 ms control rate

# learner tokens: release one every LEARN_EVERY stored transitions. SAC-ZOP stores one
# N-step transition per N=5-step MPC cycle and releases one token per store, i.e. one
# 20-step training block per 5 env steps. Plain SAC stores a transition every step, so
# gating the token to every 5th store gives the identical gradient-steps-per-env-step
# budget (4). Set to 1 for one block per env step (5x the update-to-data ratio).
LEARN_EVERY = 1

# device setup
device = "cpu"

# observation and action spaces
_x_thr = X_TERM_M  # = Arduino's own position_limit (11000 counts): host termination and
                    # hardware trip now coincide, no separate earlier threshold
_x_low = -float(_x_thr)
_x_high = float(_x_thr)

obs_low = np.array([_x_low, -np.pi, -5, -21], dtype=np.float32)
obs_high = np.array([_x_high, np.pi, 5, 21], dtype=np.float32)
obs_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

# For standard SAC, action space is the force directly
action_low = np.array([-20.0], dtype=np.float32)
action_high = np.array([20.0], dtype=np.float32)
action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

# SAC config
cfg_sac = SacTrainerConfig()

# Enable layer normalization for critic only (same as SAC-ZOP)
cfg_sac.critic_mlp.norm_layer = "layer_norm"

# collect 1000 env steps (~10 s of hardware data) before the first gradient step, like
# sim_sac.py. With train_start=0 the first updates run on a handful of near-identical
# transitions from a single episode start, which the critic overfits hard at this UTD.
cfg_sac.train_start = 200

# Replay Buffer initg
replay_buffer = ReplayBuffer(buffer_limit=cfg_sac.buffer_size, device=device)

# critic init
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

# actor init (standard SAC actor, not MPC-based)
actor = SacActor(
    extractor_cls=extractor_cls,
    observation_space=obs_space,
    action_space=action_space,
    distribution_name=cfg_sac.distribution_name,
    mlp_cfg=cfg_sac.actor_mlp,
).to(device)

# entropy temperature Alpha init
log_alpha = torch.nn.Parameter(
    torch.tensor(cfg_sac.init_alpha, dtype=torch.float32).log()
).to(device)

alpha_optimizer = (
    torch.optim.Adam([log_alpha], lr=cfg_sac.lr_alpha)
    if cfg_sac.lr_alpha is not None
    else None
)

action_dim = int(np.prod(action_space.shape))
# SAC-ZOP divides the log-prob by param_dim / action_dim because it injects noise in the
# MPC parameter space. Here the action space *is* the action, so the factor is 1.0 and the
# entropy/alpha terms are numerically identical to SAC-ZOP's.
entropy_norm = 1.0
# standard SAC heuristic for a 1-D continuous action (= -1.0), matching sim_sac.py and
# sim_sac_zop.py rather than the -2.0 in SacTrainerConfig
target_entropy = -float(action_dim)

# initializing optimizers
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg_sac.lr_q)
actor_optimizer  = torch.optim.Adam(actor.parameters(),  lr=cfg_sac.lr_pi)



def sac_update_step(batch_size, train_start):
    """
    Background thread: one training block (20 gradient steps) per learner token.
    1 semaphore token = 1 block. If tokens pile up (training slower than sampling)
    the thread runs back-to-back without waiting, using all available time.
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
                    }, step=abs_step_count)
                except Exception:
                    pass

        except Exception as e:
            print("Exception in sac_update_step:", e)


def sac_single_step_update(batch_size, update_actor=True, metrics_accumulator=None):
    """
    Pure training step: performs one critic update, plus actor + temperature updates
    on actor steps. Increments total_training_steps.
    Only collects metrics into dict - NO logging.

    Args:
        batch_size: Number of samples to use.
        update_actor: Whether to update actor this step.
        metrics_accumulator: Dict to collect metrics into, or None to skip metrics.
    """

    global total_training_steps

    if len(replay_buffer) < batch_size:
        return False

    o, a, r, o_prime, te = replay_buffer.sample(batch_size)

    # cache alpha once — avoids repeated exp() + .item() syncs
    alpha = log_alpha.exp().item()

    # critic target (no grad)
    with torch.no_grad():
        a_pi_prime, log_p_prime, _ = actor(o_prime, deterministic=False)
        q_target = target_critic(o_prime, a_pi_prime)
        q_target = torch.min(q_target, dim=1, keepdim=True).values
        factor = cfg_sac.entropy_reward_bonus / entropy_norm
        q_target = q_target - alpha * log_p_prime * factor
        target = r[:, None].to(device) + cfg_sac.gamma * (1 - te[:, None].to(device)) * q_target

    # actor forward + alpha update BEFORE critic — only on actor steps
    if update_actor:
        a_pi, log_p, _ = actor(o, deterministic=False)
        log_p = log_p / entropy_norm

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
    talk_to_arduino(0, mode=2)
    print("resetting env...")

    try:
        while True:
            state_que.get_nowait()
    except queue.Empty:
        pass

    while True:
        state = state_que.get()

        while True:
            try:
                state = state_que.get_nowait()
            except queue.Empty:
                break

        if len(state) < 5:
            continue

        x, theta, v, thetadot, tripped_flag = state

        if (not bool(tripped_flag)
            and abs(x) <= 100
            and abs(thetadot) <= 0.1
            and abs(countpersecond_to_meterspersecond(v)) <= 0.1
            and abs(theta) >= 3.1):
            break

    print(f'trying to reset. State: x={x}, theta={theta}, tripped={tripped_flag}, v={v}, thetadot={thetadot}')
    return


def load_checkpoints():
    """Load model checkpoints if a complete set exists."""
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
    """Save model checkpoints to disk."""
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


# ---- final evaluation episode -------------------------------------------------
# Run deterministic, unlogged evaluation episodes on Ctrl+C (see run_eval_episode).
# Set to False (or EVAL_ON_EXIT=0) to exit immediately instead.
RUN_EVAL_ON_EXIT = os.environ.get("EVAL_ON_EXIT", "1") not in ("0", "false", "False")

# Steps of the closing eval episode. Kept at max_ep_steps so the episodic return is
# directly comparable to the training episodes; raise it to watch the policy hold
# the pole for longer than it ever had to during training.
EVAL_MAX_STEPS = max_ep_steps

# Number of repeated closing eval episodes (same frozen policy, independent physical
# trials on hardware) run on exit.
NUM_EVAL_RUNS = 5


def run_eval_episode(max_steps: int = EVAL_MAX_STEPS, eval_run: int = 1, seed: int = 0):
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
    csv_path = os.path.join(CHECKPOINT_DIR, f"eval_sac_run{eval_run}_seed{seed}.csv")

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

    step = 0
    cum_reward = 0.0
    max_force = 0.0
    upright_streak = 0
    longest_upright = 0
    stabilized = False
    stabilized_at = None
    eval_theta_buffer = deque(maxlen=STABILIZATION_BUFFER_SIZE)
    terminated = False

    columns = [
        'step', 'time_s', 'x_counts', 'x_m', 'theta_rad', 'v_counts_s', 'v_m_s',
        'thetadot_rad_s', 'tripped', 'u_force_N', 'u_pwm', 'reward',
        'cum_reward', 'upright',
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

            with torch.no_grad():
                # deterministic=True -> mode of the distribution, no action noise
                action, _, _ = actor(obs_batch, deterministic=True)
            u_force = float(action[0].cpu().numpy().squeeze())

            u_pwm = force_to_pwm(u_force, countpersecond_to_meterspersecond(v), max_pwm_limit=150)
            max_force = max(max_force, abs(u_force))
            talk_to_arduino(u_pwm, mode=0)

            # pre-step values, paired below with the reward they produced
            row_pre = [
                step, time.perf_counter() - t0, x, counts_to_meters(x), theta,
                v, countpersecond_to_meterspersecond(v), thetadot, int(tripped),
                u_force, u_pwm,
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

            writer.writerow(row_pre + [reward, cum_reward, int(is_upright)])
            f.flush()  # crash/abort safe

            done = done_eval(state, step, max_steps, x_threshold=_x_thr)
            terminated = bool(tripped) or abs(counts_to_meters(x)) > float(_x_thr)

    except KeyboardInterrupt:
        print("\nEval episode aborted by user - partial trajectory kept.")
    finally:
        try:
            talk_to_arduino(0, mode=1)
        except Exception:
            pass
        trailer = {
            'eval_run': eval_run,
            'seed': seed,
            'deterministic': True,
            'steps': step,
            'max_steps': max_steps,
            'episodic_return': round(cum_reward, 4),
            'mean_reward_per_step': round(cum_reward / step, 4) if step else float('nan'),
            'terminated_by_trip': int(terminated),
            'stabilized': int(stabilized),
            'stabilized_at_step': stabilized_at,
            'longest_upright_steps': longest_upright,
            'upright_fraction': round(longest_upright / step, 4) if step else float('nan'),
            'max_force_N': round(max_force, 4),
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
    # seeding (same as SAC-ZOP)
    seed = 4  
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    
    load_checkpoints()  
    
    # initialize wandb (resume if we have a run_id, else create new)
    global wandb_run_id
    if wandb_run_id:
        print(f"Resuming wandb run: {wandb_run_id}")
        wandb.init(
            project="Paper-Real-SAC",
            id=wandb_run_id,
            resume="allow",
            config={
                "buffer_size": cfg_sac.buffer_size,
                "batch_size": cfg_sac.batch_size,
                "lr_q": cfg_sac.lr_q,
                "lr_pi": cfg_sac.lr_pi,
                "lr_alpha": cfg_sac.lr_alpha,
                "gamma": cfg_sac.gamma,
                "tau": cfg_sac.tau,
                "max_ep_steps": max_ep_steps,
            }
        )
    else:
        print("Starting new wandb run")
        run = wandb.init(
            project="Paper-Real-SAC",
            name=f"SAC_{int(seed)}",
            config={
                "buffer_size": cfg_sac.buffer_size,
                "batch_size": cfg_sac.batch_size,
                "lr_q": cfg_sac.lr_q,
                "lr_pi": cfg_sac.lr_pi,
                "lr_alpha": cfg_sac.lr_alpha,
                "gamma": cfg_sac.gamma,
                "tau": cfg_sac.tau,
                "max_ep_steps": max_ep_steps,
            }
        )
        wandb_run_id = run.id
        print(f"New wandb run ID: {wandb_run_id}")
    
    global episode_step_count, episode_count, abs_step_count, learning_step, current_episode_reward, episode_rewards, max_force_perep, theta_buffer, stabilized_this_episode, num_terminations, num_stabilized_steps, upright_streak, longest_upright_steps, stabilized_at_step

    # wall-clock reference for the run (used by the timing summary on exit)
    t_start = time.perf_counter()

    # start the background receiver task
    arduinoThread = threading.Thread(target=listen_to_arduino, args=())
    arduinoThread.daemon = True
    arduinoThread.start()

    # start the background learning task
    learningThread = threading.Thread(
        target=sac_update_step,
        args=(cfg_sac.batch_size, cfg_sac.train_start),
        )
    learningThread.daemon = True
    learningThread.start()
    
    # save initial checkpoints
    print("Saving initial checkpoints...")
    save_checkpoints()
    
    #Main RL Loop#################################################################
    try:
        while True:
            reset_env()
            
            talk_to_arduino(0, mode=0)
            
            # wait for the learner thread to catch up on all queued transitions
            print("Training inbetween episodes...")
            inbetween_training()

            episode_step_count = 0
            current_episode_reward = 0.0
            max_force_perep = 0
            
            # reset stabilization tracking
            theta_buffer.clear()
            stabilized_this_episode = False
            stabilized_at_step = None
            upright_streak = 0
            longest_upright_steps = 0
            
            episode_count += 1
            
            # save checkpoints at episode 1 and then every 50 episodes
            print(f"Episode {episode_count}: Checking if we should plot (episode_count % 50 = {episode_count % 50})")
            if episode_count == 1 or episode_count % 50 == 0:
                print(f"Saving checkpoints at episode {episode_count}...")
                save_checkpoints()
                
                # generate and save policy heatmaps
                print(f"Generating policy heatmaps at episode {episode_count}...")
                try:
                    actor_path = os.path.join(CHECKPOINT_DIR, 'actor.pth')
                    plot_sac_policy_heatmap(
                        actor_path=actor_path,
                        v_fixed=0.0,
                        thetadot_fixed=0.0,
                        x_range=(-_x_thr, _x_thr),
                        theta_range=(-np.pi, np.pi),
                        resolution=50,
                        plt_show=False,
                        save_path=None,
                        episode_num=episode_count
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
                            f'policy_heatmap_force_grid_ep{episode_count}.pdf',
                            f'policy_heatmap_critic_grid_ep{episode_count}.pdf',
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
                    print(f"Failed to generate/log policy heatmap: {e}")
                    import traceback
                    traceback.print_exc()
            
            state_que.queue.clear()
            
            state = None
            state = state_que.get()
            x, theta, v, thetadot, tripped = state 
            
            # clip thetadot (same as SAC-ZOP)
            thetadot = np.clip(thetadot, -20.0, 20.0)
            state = (x, theta, v, thetadot, tripped)
            
            done = done_eval(state, episode_step_count, max_ep_steps, x_threshold=_x_thr)

            # learner-token gating (see LEARN_EVERY)
            steps_since_token = 0

            print("================================")
            print("Starting new episode")
            print(f"Episode {episode_count}, Learning Step: {learning_step}, Total Steps: {abs_step_count}")

            while not done:            
                episode_step_count += 1
                abs_step_count += 1
                
                obs = sac_state_to_tensor(state, batch=False).to(device)
                obs_batch = obs.unsqueeze(0)
                
                with torch.no_grad():
                    action, log_prob, stats = actor(obs_batch, deterministic=False)

                action_t = action[0].detach().to(replay_buffer.device).float()
                u_force = float(action[0].cpu().numpy().squeeze())

                u_pwm = force_to_pwm(u_force, countpersecond_to_meterspersecond(v), max_pwm_limit=150)

                max_force_perep = max(max_force_perep, abs(u_force))

                # actuate first, then log: keeps wandb work out of the
                # observe -> act latency path
                talk_to_arduino(u_pwm, mode=0)

                # collect per-step statistics (pre-step state/action), logged in a
                # single wandb.log at the end of the iteration together with the reward
                step_stats = {
                    'step/u_force_s': u_force,
                    'step/u_pwm_s': u_pwm,
                    'step/x_s': x,
                    'step/theta_s': theta,
                    'step/v_s': v,
                    'step/thetadot_s': thetadot,
                    'step/episode_step_s': episode_step_count,
                    'step/abs_step_s': abs_step_count,
                }
                if stats:
                    for key, val in stats.items():
                        step_stats[f'step/actor_{key}_s'] = val

                state = state_que.get()
                while True:
                    try: 
                        state = state_que.get_nowait()
                    except queue.Empty:
                        break
                
                # clip thetadot (same as SAC-ZOP)
                x, theta, v, thetadot, tripped = state
                thetadot = np.clip(thetadot, -20.0, 20.0)
                state = (x, theta, v, thetadot, tripped)
                
                obs_next = sac_state_to_tensor(state, batch=False).to(device)
                
                reward = compute_reward(state, u_force)
                
                current_episode_reward += reward
                
                # track stabilization (same as SAC-ZOP)
                theta_normalized = ((theta + np.pi) % (2 * np.pi)) - np.pi
                theta_buffer.append(theta_normalized)

                theta_abs = abs(theta_normalized)
                is_upright = theta_abs <= STABILIZATION_THRESHOLD

                # longest consecutive upright run: the ungated version of the latch below
                if is_upright:
                    upright_streak += 1
                    longest_upright_steps = max(longest_upright_steps, upright_streak)
                else:
                    upright_streak = 0

                if not stabilized_this_episode and len(theta_buffer) == STABILIZATION_BUFFER_SIZE:
                    if all(abs(t) <= STABILIZATION_THRESHOLD for t in theta_buffer):
                        stabilized_this_episode = True
                        # kept for the episode log below, so it lands on the episode axis
                        stabilized_at_step = episode_step_count

                # one wandb.log per env step (reward + state/action + counters)
                try:
                    step_stats['step/reward_s'] = reward
                    step_stats['step/stabilized_s'] = int(stabilized_this_episode)
                    # raw 0/1 signal, unsmoothed: averaging it in wandb over any window
                    # gives the "fraction of time upright" curve
                    step_stats['step/upright_s'] = int(is_upright)
                    wandb.log(step_stats, step=abs_step_count)
                except Exception:
                    pass

                done = done_eval(state, episode_step_count, max_ep_steps, x_threshold=_x_thr)

                # terminated = ended by a trip (safety flag or |x| > threshold). Reaching
                # max_ep_steps is a truncation, not a termination: the value function must
                # still bootstrap there, so this - not `done` - is the buffer's terminal
                # flag (same as sim_sac.py / sim_sac_zop.py, which store int(terminated)).
                terminated = bool(tripped) or abs(counts_to_meters(x)) > float(_x_thr)

                if done:
                    talk_to_arduino(0, mode=1)
                    if terminated:
                        num_terminations += 1
                    # every step of a stabilized episode counts as stabilized (not just
                    # the ones after the latch fired). wandb steps that are already
                    # written cannot be revised, so the credit is applied here in one go.
                    if stabilized_this_episode:
                        num_stabilized_steps += episode_step_count
                    episode_rewards.append({
                        'episode': episode_count,
                        'cumulative_reward': current_episode_reward,
                        'steps': episode_step_count
                    })
                    print(f"Episode {episode_count} finished: Total Reward = {current_episode_reward:.2f}, Steps = {episode_step_count}, Max Force = {max_force_perep}, Stabilized = {stabilized_this_episode}")

                    try:
                        ep_log = {
                            # episode counter: pick this as the panel x-axis in wandb to
                            # plot any *_e metric over episodes instead of env steps
                            'episode/episode_number_e': episode_count,
                            # undiscounted sum of rewards over the episode = episodic return
                            'episode/cumulative_reward_e': current_episode_reward,
                            'episode/steps_e': episode_step_count,
                            'episode/max_force_e': max_force_perep,
                            # 1 if the pole was balanced for STABILIZATION_BUFFER_SIZE
                            # consecutive steps anywhere in this episode, else 0
                            'episode/stabilized_e': int(stabilized_this_episode),
                            'terminations/total_e': num_terminations,
                            'episode/learning_step_e': learning_step,
                            # graded version of `stabilized` - see longest_upright_steps
                            'episode/longest_upright_steps_e': longest_upright_steps,
                            # all steps of a stabilized episode, 0 otherwise
                            'episode/stabilized_steps_e': episode_step_count if stabilized_this_episode else 0,
                            'perf/stabilized_steps_total_e': num_stabilized_steps,
                        }
                        # only defined for episodes that actually stabilized
                        if stabilized_at_step is not None:
                            ep_log['stabilization/achieved_at_step_e'] = stabilized_at_step
                        wandb.log(ep_log, step=abs_step_count)
                    except Exception:
                        pass

                replay_buffer.put((obs, action_t, float(reward), obs_next, int(terminated)))

                # notify background learner (counted for drain bookkeeping).
                # gated to every LEARN_EVERY transitions so the gradient-steps-per-env-step
                # budget matches SAC-ZOP's one block per N=5-step cycle
                steps_since_token += 1
                if steps_since_token >= LEARN_EVERY or done:
                    steps_since_token = 0
                    try:
                        _note_token_released()
                    except Exception:
                        pass

    except KeyboardInterrupt:
        try:
            talk_to_arduino(0, mode=1)
            print(f'serial closed')
        except Exception:
            pass
        # print training block timing statistics (same as SAC-ZOP)
        if training_step_times:
            avg_time = np.mean(training_step_times)
            max_time = np.max(training_step_times)
            min_time = np.min(training_step_times)
            print(f"\nTraining block timing statistics (20 gradient steps per block):")
            print(f"  Total training blocks: {len(training_step_times)}")
            print(f"  Total gradient steps: {total_training_steps}")
            print(f"  Average time per block: {avg_time*1000:.2f} ms")
            print(f"  Minimum time per block: {min_time*1000:.2f} ms")
            print(f"  Maximum time per block: {max_time*1000:.2f} ms")
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

        try:
            wandb.finish()
        except Exception:
            pass
    finally:
        ser.close()
        pass



if __name__ == "__main__":
    main()

