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
num_stabilized_steps = 0  # cumulative steps spent in the balanced/stabilized mode

# stabilization tracking
STABILIZATION_BUFFER_SIZE = 40
STABILIZATION_THRESHOLD = 0.15
theta_buffer = deque(maxlen=STABILIZATION_BUFFER_SIZE)
stabilized_this_episode = False

# max steps per episode
max_ep_steps = 200  # 10 s at the 50 ms control rate

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
                        'learning/training_block_time_ms': block_elapsed_time * 1000,
                        'learning/block_step': learning_step,
                        'learning/total_training_steps': total_training_steps,
                        # backlog: tokens released but not yet processed (lock-free read)
                        'learning/pending_tokens': _tokens_released - _tokens_processed,
                        'learning/buffer_size': len(replay_buffer),
                        'learning/q_loss_avg': _safe_mean(metrics_accumulator['q_losses']),
                        'learning/pi_loss': metrics_accumulator['pi_losses'][0] if metrics_accumulator['pi_losses'] else float('nan'),
                        'learning/alpha': metrics_accumulator['alphas'][0] if metrics_accumulator['alphas'] else float('nan'),
                        'learning/q_avg': _safe_mean(metrics_accumulator['q_values']),
                        'learning/q_target_avg': _safe_mean(metrics_accumulator['q_targets']),
                        'learning/entropy': metrics_accumulator['entropies'][0] if metrics_accumulator['entropies'] else float('nan'),
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



def main():
    # seeding (same as SAC-ZOP)
    seed = 1  
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
            name=f"real_hardware_run_2_{int(seed)}",
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
    
    global episode_step_count, episode_count, abs_step_count, learning_step, current_episode_reward, episode_rewards, max_force_perep, theta_buffer, stabilized_this_episode, num_terminations, num_stabilized_steps

    # env-throughput timing (for episode/steps_per_second, like the sim scripts)
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
                    'step/u_force': u_force,
                    'step/u_pwm': u_pwm,
                    'step/x': x,
                    'step/theta': theta,
                    'step/v': v,
                    'step/thetadot': thetadot,
                    'step/episode_step': episode_step_count,
                    'step/abs_step': abs_step_count,
                }
                if stats:
                    for key, val in stats.items():
                        step_stats[f'step/actor_{key}'] = val

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
                
                if not stabilized_this_episode and len(theta_buffer) == STABILIZATION_BUFFER_SIZE:
                    if all(abs(t) <= STABILIZATION_THRESHOLD for t in theta_buffer):
                        stabilized_this_episode = True
                        num_stabilized_steps += STABILIZATION_BUFFER_SIZE  # retroactively credit the 200-step balanced window
                        step_stats['stabilization/achieved_at_step'] = episode_step_count
                        step_stats['stabilization/achieved_at_abs_step'] = abs_step_count
                        step_stats['stabilization/episode'] = episode_count
                elif stabilized_this_episode:
                    num_stabilized_steps += 1  # latched: every step after achievement counts as stabilized

                # one wandb.log per env step (reward + state/action + counters)
                try:
                    step_stats['step/reward'] = reward
                    step_stats['step/stabilized'] = int(stabilized_this_episode)
                    step_stats['stabilization/stabilized_steps_total'] = num_stabilized_steps
                    step_stats['terminations/total'] = num_terminations
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
                    elapsed = time.perf_counter() - t_start
                    sps = abs_step_count / max(elapsed, 1e-9)
                    episode_rewards.append({
                        'episode': episode_count,
                        'cumulative_reward': current_episode_reward,
                        'steps': episode_step_count
                    })
                    print(f"Episode {episode_count} finished: Total Reward = {current_episode_reward:.2f}, Steps = {episode_step_count}, Max Force = {max_force_perep}, Stabilized = {stabilized_this_episode}")

                    try:
                        wandb.log({
                            'episode/episode_number': episode_count,
                            'episode/cumulative_reward': current_episode_reward,
                            'episode/steps': episode_step_count,
                            'episode/max_force': max_force_perep,
                            'episode/stabilized': int(stabilized_this_episode),
                            'episode/steps_per_second': sps,
                            'episode/terminated': int(terminated),
                            'terminations/total': num_terminations,
                            'episode/learning_step': learning_step,
                            'episode/abs_step': abs_step_count,
                        }, step=abs_step_count)
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
        try:
            wandb.finish()
        except Exception:
            pass
    finally:
        ser.close()
        pass



if __name__ == "__main__":
    main()

