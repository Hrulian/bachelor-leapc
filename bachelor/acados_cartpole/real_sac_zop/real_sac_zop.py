import torch, threading, queue, serial, time, os, csv
import numpy as np
import gymnasium as gym
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
from bachelor.acados_cartpole.my_utils_plot import plot_cartpole_log
from leap_c.planner import ControllerFromPlanner
from leap_c.torch.nn.extractor import get_extractor_cls  # optional helper



#COMMUNICATION#####################################################################
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

# initialize step count. Used for learning updates and episode management
step_count = 0

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
_x_low = -float(_x_thr)
_x_high = float(_x_thr)

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



def sac_zop_update_step(batch_size, update_freq, soft_update_freq, train_start):
    """Background learning loop: continuously checks whether an update should run
    and performs updates when enough data / steps are available.

    This function is intended to be started in a daemon thread so it runs in the
    background while the main loop communicates with the real hardware.
    """
    global step_count
    # wait for new data via an Event to avoid busy-waiting
    # use a timeout so the thread can still periodically check for fatal conditions
    timeout_s = 1.0

    while True:
        try:
            # wait until new data is available or timeout passes
            learning_event.wait(timeout=timeout_s)
            if not learning_event.is_set():
                continue
            learning_event.clear()
            # now attempt update...

            # if woken up, attempt an update if conditions are met
            if step_count >= train_start and len(replay_buffer) >= batch_size and (step_count % update_freq == 0):
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

                # soft update
                if step_count % soft_update_freq == 0:
                    soft_target_update(critic, target_critic, cfg_saczop.tau)

        except Exception as e:
            # keep the background thread alive on unexpected errors
            import traceback

            print("Exception in sac_zop_update_step:", e)
            traceback.print_exc()
            # clear event to avoid immediate busy-loop on error
            learning_event.clear()
            time.sleep(0.1)
        finally:
            # clear the event so we wait for the next notification
            learning_event.clear()

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
        # - not tripped
        # - cart close to center (x in meters smaller than threshold)
        # - small angular velocity and cart velocity
        #print("checkinf reset condition...")
        if (not bool(tripped_flag)
            and abs(counts_to_meters(x)) <= 100
            and abs(thetadot) <= 0.0001
            and abs(countpersecond_to_meterspersecond(v)) <= 0.0):
            break
        # else: keep waiting

        print(f'env reset. State: x={x}, tripped={tripped_flag}, v={v}, thetadot={thetadot}')
    return


#MAIN LOOP###############################################################################
def main():
    # ensures we reference the module-level variables
    global ctx, step_count
    
    # max steps per episode
    max_ep_steps = 1000 # so with communication time set 10 ms -> max episode time 10s

    # init episode variables
    episode_count = 0
    
    # prepare CSV logging: remove any old log file at start so each run is fresh
    log_path = os.path.join(os.path.dirname(__file__), "real_cartpole_log.csv")
    if os.path.exists(log_path):
        try:
            os.remove(log_path)
        except Exception as e:
            print("Warning: failed to remove old log file:", e)

    log_fh = open(log_path, "a", newline="")
    log_writer = csv.writer(log_fh)
    # write header for a fresh run
    log_writer.writerow(["t", "x_m", "theta", "v_m_s", "thetadot", "u_pwm"])
    log_fh.flush()

    # start the background receiver task
    arduinoThread = threading.Thread(target=listen_to_arduino, args=())
    arduinoThread.daemon = True
    arduinoThread.start()

    # start the background learning task (use proper args)
    learningThread = threading.Thread(
        target=sac_zop_update_step,
        args=(cfg_saczop.batch_size, cfg_saczop.update_freq, cfg_saczop.soft_update_freq, cfg_saczop.train_start),
    )
    learningThread.daemon = True
    learningThread.start()
    
    
    
    #Main RL Loop#################################################################
    """
    potential issues:
        - depending on how long the learning function takes leaning steps could be skipped
        more concretely if training takes longer than the time between four env steps
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
            
            # reset ctx
            ctx = None
            
            #reset step count
            step_count = 0
            
            # per-episode bookkeeping
            episode_count += 1

            # reset state_que
            state_que.queue.clear()
            
            # wait for first valid state
            state = None
            state = state_que.get() # blocks until the thread adds a first state
            x, theta, v, thetadot, tripped = state 
            
            # check if for some reason ep is alredy done
            done = done_eval(state, step_count, max_ep_steps, x_threshold=_x_thr)
            
            
            # episode loop
            print("Starting new episode")
            while not done:            
                # step counting
                step_count += 1
                
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
                u_pwm = force_to_pwm(u_force, countpersecond_to_meterspersecond(v), 50)

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
                
                # check done 
                done = done_eval(state, step_count, max_ep_steps, x_threshold=_x_thr)   

                # store transition in replay buffer (obs, param, reward, obs_next, done)
                replay_buffer.put((obs, param_t, float(reward), obs_next, int(done)))
                # notify background learner that new data is available
                try:
                    learning_event.set()
                except Exception:
                    pass

                # # write log row
                # try:
                #     tnow = time.time()
                #     x_m = counts_to_meters(x)
                #     v_m_s = countpersecond_to_meterspersecond(v)
                #     log_writer.writerow([tnow, x_m, theta, v_m_s, thetadot, u_pwm])
                #     log_fh.flush()
                #     try:
                #         os.fsync(log_fh.fileno())
                #     except Exception:
                #         pass
                # except Exception:
                #     pass


    except KeyboardInterrupt:
        # try to stop actuator and plot
        try:
            talk_to_arduino(0, mode=1)  # tell arduino to stop
        except Exception:
            pass
        try:
            log_fh.close()
        except Exception:
            pass
        try:
            # save plot next to the CSV log with same base name
            plot_path = os.path.splitext(log_path)[0] + '.png'
            # save and show the plot so the user can inspect it interactively
            plot_cartpole_log(log_path, plt_show=True, save_path=plot_path)
            print(f'Saved and displayed plot at {plot_path}')
        except Exception as e:
            print('Plotting failed:', e)
    finally:
        try:
            if not log_fh.closed:
                log_fh.close()
        except Exception:
            pass
        ser.close()
        pass



if __name__ == "__main__":
    main()

