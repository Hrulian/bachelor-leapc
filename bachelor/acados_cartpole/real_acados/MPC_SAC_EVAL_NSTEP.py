import threading, queue, serial, time, os, csv
import torch
import numpy as np
import gymnasium as gym

from bachelor.acados_cartpole.my_helpers import (
    force_to_pwm,
    state_tuple_to_tensor,
    counts_to_meters,
    countpersecond_to_meterspersecond,
    compute_reward,
    sac_state_to_tensor,
)
from bachelor.acados_cartpole.my_planner import CartPolePlannerConfig, CartPolePlanner, create_custom_cartpole_params
from bachelor.acados_cartpole.my_utils_plot import plot_eval_log
from leap_c.planner import ControllerFromPlanner


###############################################################################
# CONFIGURATION FLAGS
###############################################################################
USE_SAC_ACTOR = True  # Set to True to use SAC-ZOP actor, False to use MPC only
N_STEPS = 5  # Call actor every N steps, use saved params for intermediate steps
CHECKPOINT_DIR = "/home/julian/Bachelor/bachelor-leapc/bachelor/acados_cartpole/real_sac_zop/checkpoints"
###############################################################################


PORT = "/dev/ttyACM0"
BAUD = 115200
FRAME_TIMEOUT_S = 0.05
MAX_PAYLOAD_LEN = 64
READ_TIMEOUT_S = 0.002

# init usb communication
ser = serial.Serial(PORT, BAUD, timeout=READ_TIMEOUT_S)
time.sleep(2)                 
ser.reset_input_buffer()  

# init the que
state_que = queue.Queue()

# init flag
ready_flag = False

# init the acados controller
cfg = CartPolePlannerConfig()
params = create_custom_cartpole_params("global", cfg.N_horizon)  # Use 'global' for SAC-ZOP
planner = CartPolePlanner(cfg, params)
controller_wrapped = ControllerFromPlanner(planner)
ctx = None

# device setup
device = "cpu"

# SAC actor (only loaded if USE_SAC_ACTOR is True)
actor = None


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
                if payload.lower() == "ready":
                    state_que.put("ready")
                else:
                    parts = payload.split(',')
                    if len(parts) == 4:
                        x, theta, v, thetadot = map(float, parts)
                        state_que.put((x, theta, v, thetadot))
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
            

def send_control(u: float):
    """
    -sends <±xx.xx> + \n 
    -newline only for debugging
    """
    ser.write(f"<{u:.3f}>\n".encode('ascii'))


def load_sac_actor():
    """Load SAC-ZOP actor from checkpoint in deterministic mode."""
    global actor
    
    try:
        from bachelor.acados_cartpole.real_sac_zop.my_sac_zop import MpcSacActor, SacZopTrainerConfig
        from leap_c.torch.nn.extractor import get_extractor_cls
        
        actor_path = os.path.join(CHECKPOINT_DIR, 'actor.pth')
        
        if not os.path.exists(actor_path):
            print(f"ERROR: Actor checkpoint not found at {actor_path}")
            print("Falling back to MPC-only mode")
            return None
        
        # Define observation and action spaces (same as in training)
        _x_thr = getattr(cfg, "x_threshold", 0.39)
        _x_low = -float(_x_thr)
        _x_high = float(_x_thr)
        obs_low = np.array([_x_low, -np.pi, -5, -21], dtype=np.float32)
        obs_high = np.array([_x_high, np.pi, 5, 21], dtype=np.float32)
        obs_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Action space is the parameter space for SAC-ZOP
        action_space = controller_wrapped.param_space
        
        # Load config
        cfg_saczop = SacZopTrainerConfig()
        cfg_saczop.critic_mlp.norm_layer = "layer_norm"
        
        # Initialize actor
        extractor_cls = get_extractor_cls("identity")
        actor = MpcSacActor(
            extractor_cls=extractor_cls,
            observation_space=obs_space,
            controller=controller_wrapped,
            distribution_name=cfg_saczop.distribution_name,
            mlp_cfg=cfg_saczop.actor_mlp,
            init_param_with_default=cfg_saczop.init_param_with_default,
        ).to(device)
        
        # Load weights
        actor.load_state_dict(torch.load(actor_path, map_location=device))
        actor.eval()
        
        print(f"Successfully loaded SAC-ZOP actor from {actor_path}")
        print("Running in DETERMINISTIC mode (no exploration)")
        return actor
        
    except Exception as e:
        print(f"ERROR loading SAC actor: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to MPC-only mode")
        return None


def main():
    global ctx, ready_flag, actor
    
    # Determine controller mode
    controller_name = "SACZOP_NSTEP" if USE_SAC_ACTOR else "MPC"
    print(f"\n{'='*60}")
    print(f"CONTROLLER MODE: {controller_name}")
    if USE_SAC_ACTOR:
        print(f"N-STEP MODE: Actor called every {N_STEPS} steps")
    print(f"{'='*60}\n")
    
    # Load SAC actor if requested
    if USE_SAC_ACTOR:
        actor = load_sac_actor()
        if actor is None:
            print("Failed to load SAC actor, exiting...")
            return
    
    # Prepare CSV logging
    log_path = os.path.join(os.path.dirname(__file__), f"eval_{controller_name.lower()}_log.csv")
    if os.path.exists(log_path):
        try:
            os.remove(log_path)
        except Exception as e:
            print("Warning: failed to remove old log file:", e)

    log_fh = open(log_path, "a", newline='')
    log_writer = csv.writer(log_fh)
    # CSV header: time, x_meters, theta_unwrapped, accumulated_reward, cycle_step, actor_called
    log_writer.writerow(["t", "x_m", "theta_unwrapped", "accumulated_reward", "cycle_step", "actor_called"])
    log_fh.flush()

    # start the background receiver task
    arduinoThread = threading.Thread(target=listen_to_arduino, args=())
    arduinoThread.daemon = True
    arduinoThread.start()
    
    # Timing and reward tracking
    control_times = []
    loop_counter = 0
    accumulated_reward = 0.0
    start_time = time.time()
    
    # Angle unwrapping: track continuous angle without -pi/pi jumps
    theta_unwrapped = 0.0
    theta_prev = None
    
    # N-step tracking
    step_in_cycle = 0
    param_current = None
    ctx_current = None
    
    # main control loop
    try:
        # wait for first valid state to initialize
        first_state = None
        while first_state is None:
            state = state_que.get()
            while True:
                try: 
                    newest_state = state_que.get_nowait()
                    state = newest_state
                except queue.Empty:
                    break
            if isinstance(state, tuple) and len(state) == 4:
                first_state = state
            else:
                print(f"DEBUG: Skipping invalid state during init: {state}")
        
        # initialize MPC solver with the first real state
        if USE_SAC_ACTOR:
            # For SAC-ZOP, convert to SAC state format
            x_init, theta_init, v_init, thetadot_init = first_state
            # Add tripped flag (0) for compute_reward compatibility
            state_with_flag = (x_init, theta_init, v_init, thetadot_init, 0)
            obs_init = sac_state_to_tensor(state_with_flag, batch=False).to(device)
            obs_init_batch = obs_init.unsqueeze(0)
            
            # Initialize planner context first
            state_converted_init = obs_init_batch
            with torch.no_grad():
                ctx_planner, _, _, _, _ = planner(state_converted_init, ctx=None)
            
            # Then initialize actor with planner context
            with torch.no_grad():
                pi_out_init = actor(obs_init_batch, ctx_planner, deterministic=True)
                ctx = pi_out_init.ctx
                ctx_current = ctx
            
            x_init, theta_init, v_init, thetadot_init = first_state
            print(f"Initialized SAC-ZOP actor with real state: x={counts_to_meters(x_init):.3f}m, theta={theta_init:.3f}rad")
        else:
            # For MPC, use standard initialization
            state_converted_init = state_tuple_to_tensor(first_state)
            ctx, _, _, _, _ = planner(state_converted_init, ctx=ctx)
            ctx_current = ctx
            x_init, theta_init, v_init, thetadot_init = first_state
            print(f"Initialized MPC solver with real state: x={counts_to_meters(x_init):.3f}m, theta={theta_init:.3f}rad")
        
        while True:
            # wait for a new state
            state = state_que.get()
            
            # drain queue to get the freshest state
            while True:
                try: 
                    newest_state = state_que.get_nowait()
                    state = newest_state
                except queue.Empty:
                    break
            
            # validate state
            if not (isinstance(state, tuple) and len(state) == 4):
                print(f"DEBUG: Skipping invalid state: {state} (type: {type(state)})")
                continue
            
            x, theta, v, thetadot = state
            
            # Unwrap theta to make it continuous (remove -pi/pi jumps)
            if theta_prev is None:
                # First iteration: initialize
                theta_unwrapped = theta
                theta_prev = theta
            else:
                # Detect wrapping: if jump is > pi, we crossed the boundary
                delta_theta = theta - theta_prev
                if delta_theta > np.pi:
                    # Wrapped from pi to -pi (going down)
                    theta_unwrapped -= (2 * np.pi - delta_theta)
                elif delta_theta < -np.pi:
                    # Wrapped from -pi to pi (going up)
                    theta_unwrapped += (2 * np.pi + delta_theta)
                else:
                    # Normal change, no wrapping
                    theta_unwrapped += delta_theta
                theta_prev = theta
            
            # Get control action based on selected controller
            t0_control = time.perf_counter()
            actor_called = False
            
            if USE_SAC_ACTOR:
                # SAC-ZOP actor with N-step pattern
                state_with_flag = (x, theta, v, thetadot, 0)
                obs = sac_state_to_tensor(state_with_flag, batch=False).to(device)
                obs_batch = obs.unsqueeze(0)
                
                # Decide whether to call actor or use existing params
                if step_in_cycle == 0:
                    # Beginning of N-step cycle: call actor to get new params
                    actor_called = True
                    
                    with torch.no_grad():
                        pi_out = actor(obs_batch, ctx_current, deterministic=True)
                    
                    # Extract and save param for next N steps
                    param_current = pi_out.param[0].detach()
                    ctx_current = pi_out.ctx
                    
                    # Get force from MPC using the parameter
                    u_force = float(pi_out.action[0].detach().cpu().numpy().squeeze())
                else:
                    # Intermediate step: use saved params with MPC planner
                    with torch.no_grad():
                        ctx_current, action, _, _, _ = planner(obs_batch, ctx_current, param_current.unsqueeze(0))
                    
                    # Get force from planner output
                    u_force = float(action[0].cpu().numpy().squeeze())
                
                # Increment cycle counter
                step_in_cycle = (step_in_cycle + 1) % N_STEPS
            else:
                # MPC controller (always calls planner)
                state_converted = state_tuple_to_tensor(state)
                ctx_current, u0, x_traj, u_traj, value = planner(state_converted, ctx=ctx_current)
                u_force = float(u0.detach().cpu().numpy().squeeze().item())
            
            elapsed = time.perf_counter() - t0_control
            ms_elapsed = round(elapsed * 1000.0, 1)
            control_times.append(ms_elapsed)
            
            # Convert force to PWM
            u_pwm = force_to_pwm(u_force, countpersecond_to_meterspersecond(v), max_pwm_limit=150)
            
            # Compute reward for this step (using helper function)
            state_with_flag = (x, theta, v, thetadot, 0)
            reward = compute_reward(state_with_flag, u_force)
            accumulated_reward += reward
            
            # Send PWM control to arduino
            send_control(u_pwm)
            
            if ready_flag == False:
                ready_flag = True
                mode_str = f"{controller_name} (N={N_STEPS})" if USE_SAC_ACTOR else controller_name
                print(f"{mode_str} controller is ready")
            
            # Write log row: time, x_meters, theta_unwrapped, accumulated_reward, cycle_step, actor_called
            try:
                tnow = time.time() - start_time
                x_m = counts_to_meters(x)
                log_writer.writerow([tnow, x_m, theta_unwrapped, accumulated_reward, step_in_cycle, int(actor_called)])
                log_fh.flush()
                os.fsync(log_fh.fileno())
            except Exception:
                import traceback
                print('Error writing to log file:')
                traceback.print_exc()
            
            loop_counter += 1
            
    except KeyboardInterrupt:
        # Stop actuator
        try:
            send_control(0.0)
        except Exception:
            pass
        try:
            log_fh.close()
        except Exception:
            pass
        
        # Print statistics
        print(f"\n{'='*60}")
        print(f"Session Summary ({controller_name} controller)")
        print(f"{'='*60}")
        print(f"Total accumulated reward: {accumulated_reward:.3f}")
        print(f"Total steps: {loop_counter}")
        
        if USE_SAC_ACTOR:
            actor_calls = loop_counter // N_STEPS + (1 if loop_counter % N_STEPS > 0 else 0)
            print(f"Actor calls: {actor_calls} (every {N_STEPS} steps)")
            print(f"MPC calls: {loop_counter - actor_calls} (intermediate steps)")
        
        # Print timing statistics
        try:
            if control_times:
                avg_time = sum(control_times) / len(control_times)
                max_time = max(control_times)
                print(f"Control computation: {len(control_times)} calls, avg={avg_time:.1f} ms, max={max_time:.1f} ms")
            else:
                print("Control timing: no samples recorded")
        except Exception:
            print("Failed to compute timing statistics")
        
        # Plot the collected data
        try:
            plot_path = os.path.splitext(log_path)[0] + '.png'
            plot_eval_log(log_path, plt_show=True, save_path=plot_path)
            print(f'\nSaved and displayed plot at {plot_path}')
        except Exception as e:
            print('Plotting failed:', e)
            import traceback
            traceback.print_exc()

    finally:
        try:
            if not log_fh.closed:
                log_fh.close()
        except Exception:
            pass
        ser.close()


if __name__ == "__main__":
    main()
