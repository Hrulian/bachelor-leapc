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
from bachelor.acados_cartpole.my_utils_plot import plot_cartpole_log
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


# TODO: make it accept rauired frames 
def listen_to_arduino():
    """
    -reads frames in the background with threading
    -frames come in the form  of <x,theta,v,thetadot,tripped>\n
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
            

def talk_to_arduino(u: int, mode: int):
    """
    -sends <±u,mode> + \n 
    -newline only for debugging
    """

    ser.write(f"<{u},{mode}>\n".encode('ascii'))




#LEARNING###############################################################################
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


# reward computation 
#TODO: fill the dummy function
def compute_reward(state, cfg_planner, mode="swingup", done=False):
    return ...

# done computation helpers
#TODO: fill the dummy function
def done_eval(a):
    if a:
        done = True
    return done

# learning thread
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




#MAIN LOOP###############################################################################
def main():
    # ensures we reference the module-level variables
    global ctx, ready_flag, step_count

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
    
    # start the background learning task
    arduinoThread = threading.Thread(target=sac_zop_update_step) #TODO: fix the arguments passed
    arduinoThread.daemon = True
    arduinoThread.start()
    

    
    #Main RL Loop#################################################################
    while True:
        # restart episode
        # TODO: implement that we wait here until arduino is ready
        talk_to_arduino(0, mode=0)  # tell arduino to reset
        
        # per-episode bookkeeping
        ep_steps += 1

        # reset state_que
        state_que.queue.clear()
        
        # wait for first valid state
        state = None
        state = state_que.get() # blocks until the thread adds a first state
        prev_obs = state_tuple_to_tensor(state).to(device)
        
        # check if for some reason ep is alreday done
        done = done_eval(state)
        
        
        # episode loop
        # TODO: find out where max_ep_steps is defined
        print("Starting new episode")
        while ep_steps < cfg_saczop.max_episode_steps and not done:
            # step counting
            step_count += 1

            # compute action/param from actor (warm-start with ctx)
            obs_batch = prev_obs.unsqueeze(0)
            with torch.no_grad():
                pi_out = actor(obs_batch, ctx, deterministic=False)

            # extract param and action (safe)
            try:
                param = pi_out.param[0].cpu().numpy()
            except Exception:
                param = None
            try:
                action_force = float(pi_out.action[0].cpu().numpy().squeeze())

            # call planner through actor to get param and action and measure time
            t0_planner = time.perf_counter()
            with torch.no_grad():
                pi_out = actor(obs_batch, ctx, deterministic=False)
            elapsed = time.perf_counter() - t0_planner
            planner_times.append(round(elapsed * 1000.0, 1))

            param = pi_out.param[0].detach().cpu().numpy()
            action_force = float(pi_out.action[0].detach().cpu().numpy().squeeze())
            ctx = pi_out.ctx
            
            # extract first control
            u_force = float(action_force.detach().cpu().numpy().squeeze().item())

            # convert first from N to PWM
            u = force_to_pwm(u_force, countpersecond_to_meterspersecond(v))

            # send PWM control to arduino
            send_control(u)
            
            if ready_flag == False:
                ready_flag = True
                print("Controller is ready")

            # wait for next hardware state (block briefly). prefer freshest reading.
            next_state = None
            t_wait_start = time.perf_counter()
            while time.perf_counter() - t_wait_start < 0.5:
                try:
                    s = state_que.get(timeout=0.5)
                except queue.Empty:
                    break
                if s == "ready":
                    continue
                if isinstance(s, tuple) and len(s) == 4:
                    # drain to freshest
                    next_state = s
                    try:
                        while True:
                            newest = state_que.get_nowait()
                            if isinstance(newest, tuple) and len(newest) == 4:
                                next_state = newest
                    except queue.Empty:
                        pass
                    break

            if next_state is None:
                # hardware timeout -> treat as terminal failure
                obs_next = state_converted
                reward = -50.0
                done = True
            else:
                obs_next = state_tuple_to_tensor(next_state)
                # compute reward and done using small helpers (swingup/balance)
                # choose mode depending on planner/environment intent
                reward = compute_reward(next_state, cfg_planner, mode="balance", done=False)
                done = (
                    abs(next_state[1]) > max_theta
                    or abs(next_state[0]) > x_thresh
                    or ep_steps + 1 >= max_ep_steps
                )

            # store transition in replay buffer (obs, param, reward, obs_next, done)
            try:
                replay_buffer.put((state_converted, param, float(reward), obs_next, bool(done)))
            except Exception:
                # fallback to numpy-friendly storage
                replay_buffer.put(
                    (
                        state_converted.detach().cpu().numpy().flatten(),
                        np.asarray(param).astype(np.float32).flatten(),
                        float(reward),
                        obs_next.detach().cpu().numpy().flatten(),
                        bool(done),
                    )
                )

            step_count += 1
            ep_steps += 1
            state = next_state if next_state is not None else state

            # minimal logging
            if step_count % 100 == 0:
                print(f"step {step_count} ep_steps={ep_steps} replay_size={len(replay_buffer)} last_reward={reward:.3f}")

            # update step (calls sac_zop_update_step defined earlier)
            sac_zop_update_step(
                batch_size=cfg_saczop.batch_size,
                update_freq=cfg_saczop.soft_update_freq if hasattr(cfg_saczop, "update_freq") is False else cfg_saczop.soft_update_freq,
                soft_update_freq=cfg_saczop.soft_update_freq,
                train_start=cfg_saczop.train_start,
            )

            # write log row
            try:
                tnow = time.time()
                x_m = counts_to_meters(x)
                v_m_s = countpersecond_to_meterspersecond(v)
                log_writer.writerow([tnow, x_m, theta, v_m_s, thetadot, u_pwm])
                log_fh.flush()
                try:
                    os.fsync(log_fh.fileno())
                except Exception:
                    pass
            except Exception:
                pass

            # episode termination handling
            if done:
                print("Episode done -> resetting context and waiting for hardware reset")
                ctx = None
                ep_steps = 0
                # give some time for hardware to settle
                time.sleep(0.5)
                # wait for a fresh valid state to start next episode
                state = None
                t0 = time.perf_counter()
                while time.time() - t0 < 5.0:
                    try:
                        s = state_que.get(timeout=5.0)
                    except queue.Empty:
                        break
                    if isinstance(s, tuple) and len(s) == 4:
                        state = s
                        break
                if state is None:
                    print("Warning: did not receive new state after episode reset")
                    # continue loop and attempt to read again

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

def main_loop_single_read(control_period_s: float = 0.05, max_steps: int = 100000):
    """
    Single-read-per-step main loop:
      - blocks for initial state
      - for each step: compute action from prev_state+ctx, send, wait for next_state, compute reward, store.
    control_period_s: desired minimum time between applied controls (>= actor+comm latency).
    """
    global ctx, step_count

    # wait for initial state
    print("Waiting for initial hardware state...")
    prev_state = None
    while prev_state is None:
        try:
            s = state_que.get(timeout=1.0)
        except queue.Empty:
            continue
        if isinstance(s, tuple) and len(s) == 4:
            prev_state = s

    prev_obs = state_tuple_to_tensor(prev_state).to(device)
    step_count = 0
    ep_steps = 0

    last_apply_time = 0.0

    while step_count < max_steps:
        t_start = time.perf_counter()

        # compute action/param from actor (warm-start with ctx)
        obs_batch = prev_obs.unsqueeze(0)
        with torch.no_grad():
            pi_out = actor(obs_batch, ctx, deterministic=False)

        # extract param and action (safe)
        try:
            param = pi_out.param[0].cpu().numpy()
        except Exception:
            param = None
        try:
            action_force = float(pi_out.action[0].cpu().numpy().squeeze())
        except Exception:
            # fallback to controller to compute u0
            with torch.no_grad():
                ctrl_out = controller(prev_obs, ctx=ctx)
                # ctrl_out expected: (param_seq, u_seq, ...)
                try:
                    u0 = ctrl_out[1]
                    action_force = float(u0.detach().cpu().numpy().squeeze())
                except Exception:
                    action_force = 0.0

        # update ctx from actor output (warm-start for next step)
        ctx = getattr(pi_out, "ctx", ctx)

        # apply action (map to PWM and send)
        u_pwm = force_to_pwm(action_force, countpersecond_to_meterspersecond(prev_state[2]))
        send_control(u_pwm)
        last_apply_time = time.perf_counter()

        # wait for new hardware state (block until next fresh tuple)
        next_state = None
        # first blocking get (gives next available frame)
        try:
            s = state_que.get(timeout=0.5)
        except queue.Empty:
            s = None

        if s == "ready":
            # ignore handshake and read again
            s = None

        if isinstance(s, tuple) and len(s) == 4:
            next_state = s
            # drain to freshest available
            try:
                while True:
                    newer = state_que.get_nowait()
                    if isinstance(newer, tuple) and len(newer) == 4:
                        next_state = newer
            except queue.Empty:
                pass

        # if no next_state arrived in timeout -> treat as failure / done
        if next_state is None:
            obs_next = prev_obs
            reward = -50.0
            done = True
        else:
            obs_next = state_tuple_to_tensor(next_state).to(device)
            reward = compute_reward(next_state, cfg_planner, mode="balance", done=False)
            # termination criteria (example)
            done = abs(next_state[1]) > 1.2 or abs(next_state[0]) > getattr(cfg_planner, "x_threshold", 0.45)

        # store transition: use param (policy param) and action_force as needed
        try:
            replay_buffer.put((prev_obs, param, float(reward), obs_next, bool(done)))
        except Exception:
            replay_buffer.put(
                (
                    prev_obs.detach().cpu().numpy().flatten(),
                    np.asarray(param).astype(np.float32).flatten() if param is not None else np.zeros(1, dtype=np.float32),
                    float(reward),
                    obs_next.detach().cpu().numpy().flatten(),
                    bool(done),
                )
            )

        step_count += 1
        ep_steps += 1
        prev_obs = obs_next
        prev_state = next_state if next_state is not None else prev_state

        # periodic learning update (non-blocking)
        sac_zop_update_step(
            batch_size=cfg_saczop.batch_size,
            update_freq=cfg_saczop.update_freq if hasattr(cfg_saczop, "update_freq") else cfg_saczop.soft_update_freq,
            soft_update_freq=cfg_saczop.soft_update_freq,
            train_start=cfg_saczop.train_start,
        )

        # maintain control cadence (sleep remaining time if actor+comm faster than desired)
        elapsed = time.perf_counter() - t_start
        sleep_for = control_period_s - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

        # handle episode termination
        if done:
            ctx = None
            ep_steps = 0
            # wait for a fresh state to start the next episode
            prev_state = None
            while prev_state is None:
                try:
                    s = state_que.get(timeout=5.0)
                except queue.Empty:
                    break
                if isinstance(s, tuple) and len(s) == 4:
                    prev_state = s
            if prev_state is None:
                print("Warning: no fresh state after episode end, continuing loop")
                # optionally break or continue depending on safety policy

    # end while
    print("Finished main loop after steps:", step_count)