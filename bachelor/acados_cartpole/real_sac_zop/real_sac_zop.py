import torch, threading, queue, serial, time, os, csv
import numpy as np
import gymnasium as gym
import wandb
from collections import deque
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
from bachelor.acados_cartpole.my_utils_plot import plot_policy_heatmap

 
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

# stabilization tracking
STABILIZATION_BUFFER_SIZE = 200  # ~1 second at 10ms sample time
STABILIZATION_THRESHOLD = 0.15  # rad, ±0.15 rad around upright
theta_buffer = deque(maxlen=STABILIZATION_BUFFER_SIZE)  # FIFO queue for theta values
stabilized_this_episode = False  # flag: was pole stabilized this episode

# max steps per episode
max_ep_steps = 1000  # Good balance between learning and hardware wear

# device setup
device = "cpu"
# MPC Layer Setup
cfg_planner = CartPolePlannerConfig()
params = create_custom_cartpole_params("global", cfg_planner.N_horizon)
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
obs_low = np.array([_x_low, -np.pi, -5, -21], dtype=np.float32)
obs_high = np.array([_x_high, np.pi, 5, 21], dtype=np.float32)
obs_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

action_space = controller_wrapped.param_space


# SacZop config
cfg_saczop = SacZopTrainerConfig()

# Enable layer normalization for critic only
cfg_saczop.critic_mlp.norm_layer = "layer_norm"

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
    With N-step buffering, this is called every N steps when a new sample is added.
    update_freq controls how many samples to collect before training once.
    Performs 20 training steps per update, with actor updated every 5th step.
    """
    
    global abs_step_count
    timeout_s = 1.0
    buffer_drops_since_update = 0  # count buffer drops since last training
    actor_update_freq = 5  # update actor every 5 critic updates

    while True:
        try:
            # wait until new data is available or timeout passes
            learning_event.wait(timeout=timeout_s)
            if not learning_event.is_set():
                continue
            learning_event.clear()

            
            # train only if we have enough samples
            if abs_step_count >= train_start:
                buffer_drops_since_update += 1
                
                # train every update_freq buffer drops (= every update_freq*N steps)
                if buffer_drops_since_update >= update_freq:
                    # perform 20 training steps
                    for i in range(20):
                        # update actor only every actor_update_freq steps
                        update_actor = (i % actor_update_freq == 0)
                        saczop_single_step_update(batch_size, update_actor=update_actor)
                    buffer_drops_since_update = 0

        except Exception as e:
            print("Exception in sac_zop_update_step:", e)
        
        finally:
            learning_event.clear()
              
              
def saczop_single_step_update(batch_size, update_actor=True):
    """
    -Performs a single SAC-ZOP update step using a batch sampled from the replay buffer.
    -Only updates when enough samples are available in the buffer.
    -also performs soft target updates at specified intervals.
    
    Args:
        batch_size: The number of samples to use for the update.
        update_actor: Whether to update the actor network in this step.
    Returns:
        A boolean indicating whether the update was performed.
    """
    
    global learning_step, training_step_times
    
    # start timing
    start_time = time.perf_counter()
    
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

    # actor update (only if update_actor is True)
    if update_actor:
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
    else:
        # compute pi_loss for logging even when not updating
        with torch.no_grad():
            pi_o = actor(o, None, only_param=True)
            a_pi = pi_o.param
            log_p = pi_o.log_prob / entropy_norm
            q_pi = torch.cat(critic(o, a_pi), dim=1)
            min_q_pi = torch.min(q_pi, dim=1, keepdim=True).values
            pi_loss = (log_alpha.exp().item() * log_p - min_q_pi).mean()
    
    # soft update targets
    if learning_step % cfg_saczop.soft_update_freq == 0:
        soft_target_update(critic, target_critic, cfg_saczop.tau)

    # increment learning step count
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
            'learning/step_time_ms': elapsed_time * 1000,  # convert to ms
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
    if wandb_run_id:
        print(f"Resuming wandb run: {wandb_run_id}")
        wandb.init(
            project="cartpole-sac-zop-1",
            id=wandb_run_id,
            resume="allow",
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
            project="cartpole-sac-zop-1",
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
    global ctx, episode_step_count, episode_count, abs_step_count, learning_step, current_episode_reward, episode_rewards, max_force_perep, theta_buffer, stabilized_this_episode
    
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
            
            # in between episode training. Train for 200 steps
            print("Training inbetween episodes...")
            inbetween_training(20)
            
            # reset ctx
            ctx = None
            
            # reset step count
            episode_step_count = 0
            
            # reset episode reward
            current_episode_reward = 0.0
            
            # reset max force tracker
            max_force_perep = 0  # Fixed variable name
            
            # reset stabilization tracking
            theta_buffer.clear()
            stabilized_this_episode = False
            
            # per-episode bookkeeping
            episode_count += 1
            
            # save checkpoints at episode 0 and then every 50 episodes
            if episode_count == 1 or episode_count % 50 == 0:
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
            N = 5  # call actor every N steps
            step_in_cycle = 0  # tracks position within N-step cycle
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
                    
                    # log actor stats
                    try:
                        step_stats = {
                            'step/u_force': u_force,
                            'step/param': param_current.cpu().numpy().tolist() if param_current.dim() > 0 else param_current.item(),
                            'step/x': x,
                            'step/theta': theta,
                            'step/v': v,
                            'step/thetadot': thetadot,
                            'step/episode_step': episode_step_count,
                            'step/cycle_step': step_in_cycle,
                            'step/actor_called': True,
                        }
                        
                        if hasattr(pi_out, 'stats') and pi_out.stats:
                            for key, val in pi_out.stats.items():
                                step_stats[f'step/pi_{key}'] = val
                        wandb.log(step_stats, step=abs_step_count)
                    except Exception:
                        pass
                
                else:
                    # intermediate step: use saved params with MPC planner
                    with torch.no_grad():
                        # call planner with saved params
                        ctx_current, action, _, _, _ = planner(obs_batch, ctx_current, param_current.unsqueeze(0))
                    
                    # get force from planner output
                    u_force = float(action[0].cpu().numpy().squeeze())
                    
                    # log planner stats
                    try:
                        step_stats = {
                            'step/u_force': u_force,
                            'step/param': param_current.cpu().numpy().tolist() if param_current.dim() > 0 else param_current.item(),
                            'step/x': x,
                            'step/theta': theta,
                            'step/v': v,
                            'step/thetadot': thetadot,
                            'step/episode_step': episode_step_count,
                            'step/cycle_step': step_in_cycle,
                            'step/actor_called': False,
                        }
                        wandb.log(step_stats, step=abs_step_count)
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
                
                # accumulate reward over N-step cycle
                accumulated_reward += reward
                
                # accumulate episode reward
                current_episode_reward += reward
                
                # track stabilization: add theta to buffer and check if stabilized
                # normalize theta to [-pi, pi] range around 0 (upright position)
                theta_normalized = ((theta + np.pi) % (2 * np.pi)) - np.pi
                theta_buffer.append(theta_normalized)
                
                # check if pole is stabilized (all recent theta values within threshold)
                if not stabilized_this_episode and len(theta_buffer) == STABILIZATION_BUFFER_SIZE:
                    if all(abs(t) <= STABILIZATION_THRESHOLD for t in theta_buffer):
                        stabilized_this_episode = True
                        # log stabilization event
                        try:
                            wandb.log({
                                'stabilization/achieved_at_step': episode_step_count,
                                'stabilization/achieved_at_abs_step': abs_step_count,
                            }, step=abs_step_count)
                        except Exception:
                            pass
                
                # log reward to wandb
                try:
                    wandb.log({'step/reward': reward}, step=abs_step_count)
                except Exception:
                    pass
                
                # check done 
                done = done_eval(state, episode_step_count, max_ep_steps, x_threshold=_x_thr)
                
                # increment cycle counter
                step_in_cycle += 1
                
                # store transition in buffer only at end of N-step cycle or if episode ends
                if step_in_cycle == N or done:
                    # store accumulated N-step transition
                    param_t = param_current.to(replay_buffer.device).float()
                    replay_buffer.put((obs_start_cycle, param_t, float(accumulated_reward), obs_next, int(done)))
                    
                    # notify background learner
                    try:
                        learning_event.set()
                    except Exception:
                        pass
                    
                    # reset cycle counter
                    step_in_cycle = 0
                
                # handle episode termination
                if done:
                    talk_to_arduino(0, mode=1)
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
                            'episode/stabilized': int(stabilized_this_episode),  # 1 if stabilized, 0 otherwise
                        }, step=abs_step_count)
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

