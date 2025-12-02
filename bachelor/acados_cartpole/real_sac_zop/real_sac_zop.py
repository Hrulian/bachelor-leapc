"""
Plan structure for real_sac_zop.py_

-MPC layer
    - as before: ocp->planner->controllerfromplanner
    
-policy in parameter space
    -use MpcSacActor from sac_zop.py
    -input: state + context file 
    -output: param vector for MPC + actual control(force) + updated context for next run
    
-parameter critic + target copy
    - use two SacCritic from sac.py
    - create target critic which is periodically updated from utils.py soft_target_update
    
-ReplayBuffer
    - use existing ReplayBuffer from buffer.py
    -store (state, param, reward, next_state, done)
    - is filled constantly from real environment interaction in my main loop
    

---------------------------------------------------------------


Main Loop Pseudocode:
    -we have previous state s, paramter p used to gnerate action a
    -apply action a to real system, get next state s', compute rewards, done(?)
    -put in buffer
    -from new state s', get new parameter p' and action a' from MpcSacActor
    -periodically call the sac_zop update (prob put in a function) 
    -if done = true ask arduino for physical reset, reset ctx and start new episode
        -done: cart to far, pole to tilted(if onyl balancing), max steps reached
        
----------------------------------------------------------------


SAC-ZOP update step
- Reuse the idea from SacZopTrainer.train_loop (sac_zop.py):
    - Sample batch from ReplayBuffer: (o, a, r, o_prime, te)
    where a = param (the parameters used to generate the last action)
    - Policy on o, o_prime in parameter space (only_param=True)
    - Temperature alpha update (optional)
    - Critic update:
    - target = r + gamma * (1 - done) * [min Q_target(o_prime, param') - alpha * log pi(param'|o_prime)]
    - critic loss = MSE(Q(o,a), target)
    - Actor update:
    - actor loss = mean(alpha * log pi(param|o) - min Q(o, param))
    - Soft-update target critic with tau
- Implement this as a standalone function:
    sac_zop_update_step()
    that uses global -,actor, -,critic -,target_critic -,log_alpha,
    optimizers, and replay buffer
"""
import torch, threading, queue, serial, time, os, csv
import numpy as np
import gymnasium as gym
from bachelor.acados_cartpole.real_sac_zop.my_sac import SacTrainerConfig, SacCritic
from bachelor.acados_cartpole.real_sac_zop.my_sac_zop import  MpcSacActor
from bachelor.acados_cartpole.real_sac_zop.my_utils import soft_target_update
from bachelor.acados_cartpole.real_sac_zop.my_buffer import ReplayBuffer
from bachelor.acados_cartpole.my_planner import CartPolePlannerConfig, CartPolePlanner, create_custom_cartpole_params
from bachelor.acados_cartpole.my_helpers import (
    force_to_pwm,
    state_tuple_to_tensor,
    counts_to_meters,
    countpersecond_to_meterspersecond,
)
from leap_c.planner import ControllerFromPlanner
from leap_c.torch.nn.mlp import MlpConfig
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

ready_flag = False # init flag


def listen_to_arduino():
    """
    -reads frames in the background with threading
    -frames come in the form  of <x,theta,v,thetadot>\n
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
                # handshake only once this branch is taken
                if payload.lower() == "ready":
                    state_que.put("ready")
                else:
                    parts = payload.split(',')
                    if len(parts) == 4:
                        # payload format: <x,theta,v,thetadot>
                        x, theta, v, thetadot = map(float, parts)
                        # store in expected order for controller: x, theta, v, thetadot
                        state_que.put((x, theta, v, thetadot))
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
            

def send_control(u: float):
    """
    -sends <±xx.xx> + \n 
    -newline only for debugging
    """
    
    ser.write(f"<{u:.3f}>\n".encode('ascii'))
    



#LEARNING INIT###############################################################################
# device setup
device = "cpu"

#initialize global variables for sac_zop update step
step_count = 0

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
cfg_saczop = SacTrainerConfig()


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
target_entropy = -action_dim if cfg_saczop.target_entropy is None else cfg_saczop.target_entropy


# initializing optimizers
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg_saczop.lr_q)
actor_optimizer  = torch.optim.Adam(actor.parameters(),  lr=cfg_saczop.lr_pi)



def sac_zop_update_step(batch_size, update_freq, soft_update_freq, train_start):
    # Learning update
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


def send_control(u: float):
    """
    -sends <±xx.xx> + \n 
    -newline only for debugging
    """
    
    ser.write(f"<{u:.3f}>\n".encode('ascii'))
    

def main():
    # plotting is provided by `plot_cartpole_log` in `utils.py`

    # ensures we reference the module-level variables
    global ctx  
    global ready_flag
    
    
    # prepare CSV logging: remove any old log file at start so each run is fresh
    log_path = os.path.join(os.path.dirname(__file__), "real_cartpole_log.csv")
    if os.path.exists(log_path):
        try:
            os.remove(log_path)
        except Exception as e:
            print("Warning: failed to remove old log file:", e)

    log_fh = open(log_path, "a", newline='')
    log_writer = csv.writer(log_fh)
    # write header for a fresh run
    log_writer.writerow(["t", "x_m", "theta", "v_m_s", "thetadot", "u_pwm"])
    log_fh.flush()


    # start the background receiver task
    arduinoThread = threading.Thread(target=listen_to_arduino, args=())
    arduinoThread.daemon = True
    arduinoThread.start()
    # collect planner timing statistics
    planner_times = []
    # debug logging timer (lightweight)
    last_dbg_time = 0.0
    
    
    
    # main control loop
    try:
        while True:
            # wait for a new state
            state = state_que.get() # blocks untill the thread adds an item
            
            while True:
                try: 
                    state = state_que.get_nowait()
                
                except queue.Empty:
                    break
            
            # just in case something else is in the que
            if not (isinstance(state, tuple) and len(state) == 4):
                continue
            
            # we finally got the newest state and extract it now
            x, theta, v, thetadot = state 

            # prepare state as torch tensor for the planner
            state_converted = state_tuple_to_tensor(state)
            
            # call planner: returns (ctx, u0, x_traj, u_traj, value)
            t0_planner = time.perf_counter()
            ctx, u0, x_traj, u_traj, value = controller(state_converted, ctx=ctx)
            elapsed = time.perf_counter() - t0_planner
            # store planner time in milliseconds with one decimal place
            ms_elapsed = round(elapsed * 1000.0, 1)
            planner_times.append(ms_elapsed)
            
            # extract first control
            u_force = float(u0.detach().cpu().numpy().squeeze().item())

            # convert first from N to PWM
            u = force_to_pwm(u_force, countpersecond_to_meterspersecond(v) )

            # send PWM control to arduino
            send_control(u)
            
            if ready_flag == False:
                ready_flag = True
                print("Controller is ready")            
            
            #debug: print the state vector (converted) immediately after sending control
    
            # try:
            #     x_m_dbg = counts_to_meters(x)
            #     v_m_s_dbg = countpersecond_to_meterspersecond(v)
            #     # original theta from the incoming state tuple
            #     theta_orig = float(state[1])

            #     # try to extract theta from the converted tensor/array
            #     try:
            #         # common case: torch tensor -> support .item()
            #         theta_conv = float(state_converted[1].item())
            #     except Exception:
            #         try:
            #             # fall back to numpy extraction if tensor has detach
            #             theta_conv = float(state_converted.detach().cpu().numpy().flatten()[1])
            #         except Exception:
            #             # final fallback: use original theta
            #             theta_conv = theta_orig

            #     print(
            #         f"theta_conv={theta_conv:.4f}, theta_orig={theta_orig:.4f}: "
            #         f"x_m={x_m_dbg:.4f}, v_m_s={v_m_s_dbg:.4f}, force={u_force:.4f}, u={u:.3f}"
            #     )
            # except Exception:
            #     print("DEBUG: failed to compute debug state")

            #write a minimal log row. Needed for plotting later
            try:
                tnow = time.time()
                x_m = counts_to_meters(x)
                v_m_s = countpersecond_to_meterspersecond(v)
                log_writer.writerow([tnow, x_m, theta, v_m_s, thetadot, u])
                # periodic lightweight debug print (once per 0.1s)
                # try:
                #     if tnow - last_dbg_time >= 0.1:
                #         last_dbg_time = tnow
                #         print(f"DBG t={tnow:.1f} x_m={x_m:.4f} theta={theta:.4f} v_m_s={v_m_s:.4f} u_force={u_force:.4f} pwm={u}")
                # except Exception:
                #     pass
                # ensure data is written to disk (helps when plotting after abrupt stops)
                try:
                    log_fh.flush()
                    os.fsync(log_fh.fileno())
                except Exception:
                    # flushing is best-effort; if it fails, keep running but print debug
                    print('Warning: failed to fsync log file')
            except Exception:
                # keep running even if logging fails; print exception to help debugging
                import traceback
                print('Error writing to log file:')
                traceback.print_exc()
            
            
    except KeyboardInterrupt:
        # try to stop actuator and plot
        try:
            send_control(0.0)
        except Exception:
            pass
        try:
            log_fh.close()
        except Exception:
            pass
        # plot the collected data
        try:
            # save plot next to the CSV log with same base name
            plot_path = os.path.splitext(log_path)[0] + '.png'
            # save and show the plot so the user can inspect it interactively
            plot_cartpole_log(log_path, plt_show=True, save_path=plot_path)
            print(f'Saved and displayed plot at {plot_path}')
        except Exception as e:
            print('Plotting failed:', e)
        # print planner timing statistics
        try:
            if planner_times:
                avg_time = sum(planner_times) / len(planner_times)
                max_time = max(planner_times)
                print(f"Planner calls: {len(planner_times)}, avg={avg_time:.1f} ms, max={max_time:.1f} ms")
                print(planner_times)
            else:
                print("Planner timing: no samples recorded")
        except Exception:
            print("Failed to compute planner timing statistics")

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