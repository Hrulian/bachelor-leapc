import torch, threading, queue, serial, time, os, csv
import numpy as np
import gymnasium as gym
import wandb
from bachelor.acados_cartpole.real_sac_zop.my_sac import SacCritic
from bachelor.acados_cartpole.real_sac_zop.my_sac_zop import  MpcSacActor, SacZopTrainerConfig
from bachelor.acados_cartpole.real_sac_zop.my_utils import soft_target_update
from bachelor.acados_cartpole.real_sac_zop.my_buffer import ReplayBuffer
from bachelor.acados_cartpole.my_planner import (
    CartPolePlannerConfig, 
    CartPolePlanner, 
    create_custom_cartpole_params
)
from bachelor.acados_cartpole.my_helpers import (
    force_to_pwm,
    countpersecond_to_meterspersecond,
    counts_to_meters,
    compute_reward,
    done_eval,
    sac_state_to_tensor,
)
 
from leap_c.planner import ControllerFromPlanner
from leap_c.torch.nn.extractor import get_extractor_cls  # optional helper



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
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


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
# event to notify learner thread of new data
learning_event = threading.Event()  

# initialize counters
episode_step_count = 0 # steps in current episode
learning_step = 0   # total learning updates performed
episode_count = 0   # total episodes completed
abs_step_count = 0  # total steps across all episodes

# episode reward tracking
episode_rewards = []  # list of cumulative rewards per episode
current_episode_reward = 0.0  # accumulated reward in current episode
max_force_perep = 0  # track max force per episode
wandb_run_id = None  # wandb run ID for resuming runs

# max steps per episode
max_ep_steps = 1000 # so with communication time set 10 ms -> max episode time 10s

# device setup
device = "cpu"

# MPC Layer Setup
cfg_planner = CartPolePlannerConfig()
params = create_custom_cartpole_params("stagewise", cfg_planner.N_horizon)
planner = CartPolePlanner(cfg_planner, params)
controller_wrapped = ControllerFromPlanner(planner)
ctx = None


# observation and action spaces
# state is (x, theta, xdot, thetadot)
# use planner config for reasonable x-bounds and clamp angle to [-pi, pi]
_x_thr = getattr(cfg_planner, "x_threshold", None)
_x_low = -float(_x_thr) # in meters
_x_high = float(_x_thr) # in meters

#TODO: double check if these spaces are correct 
obs_low = np.array([_x_low, -np.pi, -np.inf, -np.inf], dtype=np.float32)
obs_high = np.array([_x_high, np.pi, np.inf, np.inf], dtype=np.float32)
obs_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

action_space = controller_wrapped.param_space


# SacZop config
cfg_saczop = SacZopTrainerConfig()


# Replay Buffer init
replay_buffer = ReplayBuffer(buffer_limit=cfg_saczop.buffer_size, device=device)


# critic init
# TODO: really no idea what that it. Check later
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
target_entropy = -action_dim if cfg_saczop.target_entropy is None else cfg_saczop.target_entropy


# initializing optimizers
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg_saczop.lr_q)
actor_optimizer  = torch.optim.Adam(actor.parameters(),  lr=cfg_saczop.lr_pi)



def sac_zop_update_step(batch_size, update_freq, train_start):
    """
    Background thread function to perform SAC-ZOP updates at specified intervals.
    """
    
    global abs_step_count
    timeout_s = 1.0

    while True:
        try:
            # wait until new data is available or timeout passes
            learning_event.wait(timeout=timeout_s)
            if not learning_event.is_set():
                continue
            learning_event.clear()

            
            if (abs_step_count >= train_start and
                abs_step_count % update_freq == 0):
                
                saczop_single_step_update(batch_size)

        except Exception as e:
            print("Exception in sac_zop_update_step:", e)
        
        finally:
            learning_event.clear()
              
              
def saczop_single_step_update(batch_size):
    """
    -Performs a single SAC-ZOP update step using a batch sampled from the replay buffer.
    -Only updates when enough samples are available in the buffer.
    -also performs soft target updates at specified intervals.
    
    Args:
        batch_size: The number of samples to use for the update.
    Returns:
        A boolean indicating whether the update was performed.
    """
    
    global learning_step
    
    # if woken up, attempt an update if conditions are met
    if len(replay_buffer) < batch_size:
        return False
    
    # sample batch
    o, a, r, o_prime, te = replay_buffer.sample(batch_size)

    # policy params for o and o_prime
    with torch.no_grad():
        pi_o_prime = actor(o_prime, None, only_param=True)
        q_target = torch.cat(target_critic(o_prime, pi_o_prime.param), dim=1)
        q_target = torch.min(q_target, dim=1, keepdim=True).values

        factor = cfg_saczop.entropy_reward_bonus / entropy_norm
        q_target = q_target - (log_alpha.exp().item()) * pi_o_prime.log_prob * factor

        target = r[:, None].to(device) + cfg_saczop.gamma * (1 - te[:, None].to(device)) * q_target

    # critic update
    q = torch.cat(critic(o, a), dim=1)
    q_loss = torch.mean((q - target).pow(2))

    critic_optimizer.zero_grad()
    q_loss.backward()
    critic_optimizer.step()

    # actor update
    pi_o = actor(o, None, only_param=True)
    a_pi = pi_o.param
    log_p = pi_o.log_prob / entropy_norm

    # temperature update
    if alpha_optimizer is not None:
        alpha_loss = -torch.mean(log_alpha.exp() * (log_p + target_entropy).detach())
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()

    q_pi = torch.cat(critic(o, a_pi), dim=1)
    min_q_pi = torch.min(q_pi, dim=1, keepdim=True).values
    pi_loss = (log_alpha.exp().item() * log_p - min_q_pi).mean()

    actor_optimizer.zero_grad()
    pi_loss.backward()
    actor_optimizer.step()
    
    # soft update targets
    if learning_step % cfg_saczop.soft_update_freq == 0:
        soft_target_update(critic, target_critic, cfg_saczop.tau)

    # increment learning step count
    learning_step += 1
    
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
        ok = saczop_single_step_update(cfg_saczop.batch_size)
        if not ok:
            # not enough data in buffer yet
            break  


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
            and abs(countpersecond_to_meterspersecond(v)) <= 0.0    # cart not moving
            and abs(theta) >= 3.0):                                # pole down
                
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
    # load checkpoints/trained models if available
    # this means training progress persists across restarts
    # it consists of actor, critic, target_critic, log_alpha + meta info. Not the buffer
    load_checkpoints()  
    
    # initialize wandb (resume if we have a run_id, else create new)
    global wandb_run_id
    if wandb_run_id:
        print(f"Resuming wandb run: {wandb_run_id}")
        wandb.init(
            project="cartpole-sac-zop",
            id=wandb_run_id,
            resume="must",
            config={
                "buffer_size": cfg_saczop.buffer_size,
                "batch_size": cfg_saczop.batch_size,
                "lr_q": cfg_saczop.lr_q,
                "lr_pi": cfg_saczop.lr_pi,
                "lr_alpha": cfg_saczop.lr_alpha,
                "gamma": cfg_saczop.gamma,
                "tau": cfg_saczop.tau,
                "max_ep_steps": max_ep_steps,
            }
        )
    else:
        print("Starting new wandb run")
        run = wandb.init(
            project="cartpole-sac-zop",
            name=f"real_hardware_run_{int(time.time())}",
            config={
                "buffer_size": cfg_saczop.buffer_size,
                "batch_size": cfg_saczop.batch_size,
                "lr_q": cfg_saczop.lr_q,
                "lr_pi": cfg_saczop.lr_pi,
                "lr_alpha": cfg_saczop.lr_alpha,
                "gamma": cfg_saczop.gamma,
                "tau": cfg_saczop.tau,
                "max_ep_steps": max_ep_steps,
            }
        )
        wandb_run_id = run.id
        print(f"New wandb run ID: {wandb_run_id}")
    
    # ensures we reference the module-level variables
    global ctx, episode_step_count, episode_count, abs_step_count, learning_step, current_episode_reward, episode_rewards, max_episode_pwm
    
    # start the background receiver task
    arduinoThread = threading.Thread(target=listen_to_arduino, args=())
    arduinoThread.daemon = True
    arduinoThread.start()

    # start the background learning task (use proper args)
    learningThread = threading.Thread(
        target=sac_zop_update_step,
        args=(
            cfg_saczop.batch_size, 
            cfg_saczop.update_freq, 
            cfg_saczop.train_start),
        )
    learningThread.daemon = True
    learningThread.start()
    
    
    
    #Main RL Loop#################################################################
    """
    potential issues:
        - depending on how long the learning function takes leaning steps could be skipped
        more concretely: if training takes longer than the time between four env steps
        -currently the controller is initialized with x0 = (0,-pi,0,0) all the time
        this is not the real state so the first MPC call might not the best but yolo
        - so the buffer is locked now and i think mostly safe in terms of threading
        - i think the actor should be thread safe aswell prob train a copy of the actor
        in the background thread and update actormain -> copyactor periodically?
        """
    try:
        while True:
            # restart episode. Wait until env is reseted
            reset_env()
            
            # env is reseted so set new mode
            talk_to_arduino(0, mode=0)  # -> arduino is ready for normal operation
            
            # in between episode training. Train for 50 steps
            print("Training inbetween episodes...")
            inbetween_training(50)
            
            # reset ctx
            ctx = None
            
            # reset step count
            episode_step_count = 0
            
            # reset episode reward
            current_episode_reward = 0.0
            
            # reset max PWM tracker
            max_force_per_ep = 0
            
            # per-episode bookkeeping
            episode_count += 1

            # reset state_que
            state_que.queue.clear()
            
            # wait for first valid state
            state = None
            state = state_que.get() # blocks until the thread adds a first state
            x, theta, v, thetadot, tripped = state 
            
            # check if for some reason ep is alredy done
            done = done_eval(state, episode_step_count, max_ep_steps, x_threshold=_x_thr)
            
            
            # episode loop
            print("================================")
            # print(f"start state: x={x}, theta={theta}, v={v}, thetadot={thetadot}, tripped={tripped}")
            print("Starting new episode")
            print(f"Episode {episode_count}, Learning Step: {learning_step}, Total Steps: {abs_step_count}")

            while not done:            
                # step counting
                episode_step_count += 1
                abs_step_count += 1
                
                # make state ready for buffer and actor
                obs = sac_state_to_tensor(state, batch=False).to(device)

                # compute action/param from actor (warm-start with ctx)
                # TODO: find out what detreministic does here
                obs_batch = obs.unsqueeze(0)
                with torch.no_grad():
                    pi_out = actor(obs_batch, ctx, deterministic=False)

                # extract param and action
                # keep param as a tensor (move to replay buffer device) so collate works
                param_t = pi_out.param[0].detach().to(replay_buffer.device).float()
                u_force = float(pi_out.action[0].cpu().numpy().squeeze())

                # convert first from N to PWM
                # TODO: fix that function currently friction still included
                u_pwm = force_to_pwm(u_force, countpersecond_to_meterspersecond(v), 255)
                
                # track maximum absolute PWM
                max_force_perep = max(u_force, abs(u_force))
                
                # log per-step statistics to wandb
                try:
                    step_stats = {
                        'step/u_force': u_force,
                        'step/u_pwm': u_pwm,
                        'step/param': param_t.cpu().numpy().tolist() if param_t.dim() > 0 else param_t.item(),
                        'step/x': x,
                        'step/theta': theta,
                        'step/v': v,
                        'step/thetadot': thetadot,
                        'step/episode_step': episode_step_count,
                    }
                    # add pi_output stats if available
                    if hasattr(pi_out, 'stats') and pi_out.stats:
                        for key, val in pi_out.stats.items():
                            step_stats[f'step/pi_{key}'] = val
                    wandb.log(step_stats, step=abs_step_count)
                except Exception:
                    pass

                # send PWM control to arduino
                talk_to_arduino(u_pwm, mode=0)
                
                # wait for next state and drain que to freshest
                state = state_que.get() # blocks untill the thread adds an item
                while True:
                    try: 
                        state = state_que.get_nowait()
    
                    except queue.Empty:
                        break
                
                # make the state ready for buffer (unbatched tensor on device)
                obs_next = sac_state_to_tensor(state, batch=False).to(device)
                x, theta, v, thetadot, tripped = state # all the quantities are not converted yet
                
                # compute reward
                reward = compute_reward(state)
                
                # accumulate episode reward
                current_episode_reward += reward
                
                # log reward to wandb
                try:
                    wandb.log({'step/reward': reward}, step=abs_step_count)
                except Exception:
                    pass
                
                # check done 
                done = done_eval(state, episode_step_count, max_ep_steps, x_threshold=_x_thr)
                
                # double check  for safety
                if done:
                    talk_to_arduino(0, mode=1)
                    # save episode reward when episode ends
                    episode_rewards.append({
                        'episode': episode_count,
                        'cumulative_reward': current_episode_reward,
                        'steps': episode_step_count
                    })
                    print(f"Episode {episode_count} finished: Total Reward = {current_episode_reward:.2f}, Steps = {episode_step_count}, Max Force = {max_force_perep}")
                    
                    # log episode statistics to wandb
                    try:
                        wandb.log({
                            'episode/episode_number': episode_count,
                            'episode/cumulative_reward': current_episode_reward,
                            'episode/steps': episode_step_count,
                            'episode/max_force': max_force_perep,
                        }, step=abs_step_count)
                    except Exception:
                        pass

                # store transition in replay buffer (obs, param, reward, obs_next, done)
                replay_buffer.put((obs, param_t, float(reward), obs_next, int(done)))
                # notify background learner that new data is available
                try:
                    learning_event.set()
                except Exception:
                    pass
                

    except KeyboardInterrupt:
        # try to stop actuator and plot
        try:
            talk_to_arduino(0, mode=1)  # tell arduino to stop
            print(f'serial closed')
        except Exception:
            pass
        # save model checkpoints on exit so training progress persists
        try:
            save_checkpoints()
        except Exception:
            pass
        # finish wandb run
        try:
            wandb.finish()
        except Exception:
            pass
    finally:
        ser.close()
        pass



if __name__ == "__main__":
    main()

