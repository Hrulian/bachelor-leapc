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
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

# timing tracking for training step
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
learning_event = threading.Event()  

# initialize counters
episode_step_count = 0
learning_step = 0
episode_count = 0
abs_step_count = 0

# episode reward tracking
episode_rewards = []
current_episode_reward = 0.0
max_force_perep = 0
wandb_run_id = None

# metrics mirrored from the sim scripts (sim_sac / sim_sac_zop / sim_sac_zopfill)
num_terminations = 0  # cumulative episodes that ended in a trip (terminated, not truncated)
num_stabilized_steps = 0  # cumulative steps spent in the balanced/stabilized mode

# stabilization tracking
STABILIZATION_BUFFER_SIZE = 200
STABILIZATION_THRESHOLD = 0.15
theta_buffer = deque(maxlen=STABILIZATION_BUFFER_SIZE)
stabilized_this_episode = False

# max steps per episode
max_ep_steps = 1000

# device setup
device = "cpu"

# observation and action spaces
_x_thr = 0.4
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

# Replay Buffer init
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
target_entropy = -action_dim if cfg_sac.target_entropy is None else cfg_sac.target_entropy

# initializing optimizers
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg_sac.lr_q)
actor_optimizer  = torch.optim.Adam(actor.parameters(),  lr=cfg_sac.lr_pi)



def sac_update_step(batch_size, update_freq, train_start):
    """
    Background thread function to perform SAC updates at specified intervals.
    Performs 20 training steps per update, with actor updated every 5th step.
    """
    
    global abs_step_count
    timeout_s = 1.0
    buffer_drops_since_update = 0
    actor_update_freq = 5  # update actor every 5 critic updates

    while True:
        try:
            learning_event.wait(timeout=timeout_s)
            if not learning_event.is_set():
                continue
            learning_event.clear()

            # train only if we have enough samples
            if abs_step_count >= train_start:
                buffer_drops_since_update += 1
                
                # train every update_freq buffer drops
                if buffer_drops_since_update >= update_freq:
                    # perform 20 training steps
                    for i in range(20):
                        # update actor only every actor_update_freq steps
                        update_actor = (i % actor_update_freq == 0)
                        sac_single_step_update(batch_size, update_actor=update_actor)
                    buffer_drops_since_update = 0

        except Exception as e:
            print("Exception in sac_update_step:", e)
        
        finally:
            learning_event.clear()


def sac_single_step_update(batch_size, update_actor=True):
    """
    Performs a single SAC update step using a batch sampled from the replay buffer.
    
    Args:
        batch_size: The number of samples to use for the update.
        update_actor: Whether to update the actor network in this step.
    Returns:
        A boolean indicating whether the update was performed.
    """
    
    global learning_step, training_step_times
    
    # start timing
    start_time = time.perf_counter()
    
    if len(replay_buffer) < batch_size:
        return False
    
    # sample batch
    o, a, r, o_prime, te = replay_buffer.sample(batch_size)

    # sample action from current policy (needed for both temperature and actor updates)
    a_pi, log_p, _ = actor(o, deterministic=False)

    # temperature update (must happen first, using current policy samples)
    if update_actor and alpha_optimizer is not None:
        alpha_loss = -torch.mean(log_alpha.exp() * (log_p + target_entropy).detach())
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()

    # update critic
    alpha = log_alpha.exp().item()
    with torch.no_grad():
        a_pi_prime, log_p_prime, _ = actor(o_prime, deterministic=False)
        q_target = torch.cat(target_critic(o_prime, a_pi_prime), dim=1)
        q_target = torch.min(q_target, dim=1, keepdim=True).values

        # subtract entropy term (standard SAC)
        q_target = q_target - alpha * log_p_prime * float(cfg_sac.entropy_reward_bonus)

        target = r[:, None].to(device) + cfg_sac.gamma * (1 - te[:, None].to(device)) * q_target

    q = torch.cat(critic(o, a), dim=1)
    q_loss = torch.mean((q - target).pow(2))

    critic_optimizer.zero_grad()
    q_loss.backward()
    critic_optimizer.step()

    # actor update (only if update_actor is True)
    if update_actor:
        q_pi = torch.cat(critic(o, a_pi), dim=1)
        min_q_pi = torch.min(q_pi, dim=1, keepdim=True).values
        pi_loss = (alpha * log_p - min_q_pi).mean()

        actor_optimizer.zero_grad()
        pi_loss.backward()
        actor_optimizer.step()
    else:
        # compute pi_loss for logging even when not updating
        with torch.no_grad():
            q_pi = torch.cat(critic(o, a_pi), dim=1)
            min_q_pi = torch.min(q_pi, dim=1, keepdim=True).values
            pi_loss = (alpha * log_p - min_q_pi).mean()
    
    # soft update targets
    if learning_step % cfg_sac.soft_update_freq == 0:
        soft_target_update(critic, target_critic, cfg_sac.tau)

    learning_step += 1
    
    # record timing
    elapsed_time = time.perf_counter() - start_time
    training_step_times.append(elapsed_time)
    
    # log learning statistics to wandb
    try:
        wandb.log({
            'learning/q_loss': q_loss.item(),
            'learning/pi_loss': pi_loss.item(),
            'learning/alpha': log_alpha.exp().item(),
            'learning/q': q.mean().item(),
            'learning/q_target': target.mean().item(),
            'learning/entropy': -log_p.mean().item(),
            'learning/learning_step': learning_step,
            'learning/step_time_ms': elapsed_time * 1000,
            'learning/buffer_size': len(replay_buffer),
        }, step=abs_step_count)
    except Exception:
        pass
    
    return True


def inbetween_training(num_updates: int):
    """
    Perform additional training updates between episodes.
    Args:
        num_updates: Number of update steps to perform.
    """
    
    for _ in range(num_updates):
        ok = sac_single_step_update(cfg_sac.batch_size)
        if not ok:
            break  


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
    seed = 0  
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
            project="cartpole-sac-1",
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
            project="cartpole-sac-1",
            name=f"real_hardware_run_{int(time.time())}",
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
        args=(
            cfg_sac.batch_size, 
            cfg_sac.update_freq, 
            cfg_sac.train_start),
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
            
            print("Training inbetween episodes...")
            inbetween_training(20)
            
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
                
                # log per-step statistics (complete logging like SAC-ZOP)
                try:
                    step_stats = {
                        'step/u_force': u_force,
                        'step/u_pwm': u_pwm,
                        'step/action': action_t.cpu().numpy().tolist() if action_t.dim() > 0 else action_t.item(),
                        'step/x': x,
                        'step/x_meters': counts_to_meters(x),
                        'step/theta': theta,
                        'step/theta_deg': np.degrees(theta),
                        'step/v': v,
                        'step/v_mps': countpersecond_to_meterspersecond(v),
                        'step/thetadot': thetadot,
                        'step/episode_step': episode_step_count,
                        'step/abs_step': abs_step_count,
                        'step/stabilized': int(stabilized_this_episode),
                        'stabilization/stabilized_steps_total': num_stabilized_steps,
                        'terminations/total': num_terminations,
                    }
                    if stats:
                        for key, val in stats.items():
                            step_stats[f'step/actor_{key}'] = val
                    wandb.log(step_stats, step=abs_step_count)
                except Exception:
                    pass

                talk_to_arduino(u_pwm, mode=0)
                
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
                        try:
                            wandb.log({
                                'stabilization/achieved_at_step': episode_step_count,
                                'stabilization/achieved_at_abs_step': abs_step_count,
                                'stabilization/episode': episode_count,
                            }, step=abs_step_count)
                        except Exception:
                            pass
                elif stabilized_this_episode:
                    num_stabilized_steps += 1  # latched: every step after achievement counts as stabilized
                
                try:
                    wandb.log({'step/reward': reward}, step=abs_step_count)
                except Exception:
                    pass
                
                done = done_eval(state, episode_step_count, max_ep_steps, x_threshold=_x_thr)
                
                if done:
                    talk_to_arduino(0, mode=1)
                    # terminated = ended by a trip (safety flag or |x| > threshold);
                    # otherwise the episode was truncated (max_ep_steps reached)
                    terminated = bool(tripped) or abs(counts_to_meters(x)) > float(_x_thr)
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

                replay_buffer.put((obs, action_t, float(reward), obs_next, int(done)))
                
                try:
                    learning_event.set()
                except Exception:
                    pass

    except KeyboardInterrupt:
        try:
            talk_to_arduino(0, mode=1)
            print(f'serial closed')
        except Exception:
            pass
        # print training step timing statistics (same as SAC-ZOP)
        if training_step_times:
            avg_time = np.mean(training_step_times)
            max_time = np.max(training_step_times)
            min_time = np.min(training_step_times)
            print(f"\nTraining step timing statistics:")
            print(f"  Total training steps: {len(training_step_times)}")
            print(f"  Average time per step: {avg_time*1000:.2f} ms")
            print(f"  Minimum time per step: {min_time*1000:.2f} ms")
            print(f"  Maximum time per step: {max_time*1000:.2f} ms")
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

