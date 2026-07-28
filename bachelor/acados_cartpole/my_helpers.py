import numpy as np
import torch

from bachelor.acados_cartpole.simulation_zaczop.rewards import get_reward_fn

# Reward used on hardware, by name from simulation_zaczop/rewards.py. Keep this in sync
# with the --reward default of the sim scripts: delegating to the shared registry instead
# of re-implementing the formula here is what stops sim and hardware from drifting apart
# (they previously disagreed by a +0.5 upright bonus). "default" is the old formula.
HARDWARE_REWARD = "cos_bonus_spin"
_reward_fn = get_reward_fn(HARDWARE_REWARD)

# Episode termination threshold [m], shared by all hardware scripts so they cannot drift
# apart. The host ends the episode (and marks the transition terminal) at this position;
# the Arduino's own safety trip stays at position_limit = 11000 counts = 0.430 m and is
# only a backstop, i.e. `tripped` should never fire in normal operation. Margin between
# the two: 0.050 m (1272 counts) ~ 2-3 control steps at full cart speed.
X_TERM_M = 0.38


def force_to_pwm(force: float, velocity: float, max_pwm_limit) -> int:
    """
    Map a desired force and current velocity to a PWM integer.

    Parameters
    - force: desired force in Newtons [N].
    - velocity: current cart velocity in meters per second [m/s].
    - max_pwm_limit: application cap for absolute PWM (default 255).

    Returns
    - pwm: integer PWM command in range [-max_pwm_limit, max_pwm_limit].
    """
    # Estimated parameters: a=48.457122, b=5.280475, c=-7.571013
    a = 48.45
    b = 5.28
    c = -7.57
    mass = 0.1744  

    # compute control-like quantity u 
    u = ((force / mass) + a * velocity - c * sgn(velocity)) / b
    #print(u)
    # map u in [-max_u, max_u] to PWM in [-255, 255]
    max_u = 24.0
    pwm_full = (u / max_u) * 255.0

    # saturate to hardware full-scale first (safe)
    pwm_full = max(-255.0, min(255.0, pwm_full))

    # round and then apply application cap
    pwm = int(round(pwm_full))
    pwm = max(-max_pwm_limit, min(max_pwm_limit, pwm))
    return pwm



    

def sgn(x: float) -> int:
    """
    Sign function.

    Parameters
    - x: numeric input

    Returns
    - -1 if x < 0, 1 if x > 0, 0 if x == 0
    """
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def differentiable_signum(x) -> float:
    """
    Smooth approximation of the signum function.
    """

    return (2 / (1 + np.exp(-10 * x))) - 1



def counts_to_meters(counts: int) -> float:
    """
    Convert encoder counts to linear position in meters.

    Parameters
    - counts: encoder counts (integer).

    Returns
    - position in meters [m].
    """
    counts_per_revolution = 1024  
    wheel_circumference = 0.04  
    revolutions = counts / counts_per_revolution
    position = revolutions * wheel_circumference
    return position


def countpersecond_to_meterspersecond(counts_per_second: int) -> float:
    """
    Convert encoder counts-per-second to linear velocity in meters per second.

    Parameters
    - counts_per_second: encoder counts per second (integer).

    Returns
    - velocity in meters per second [m/s].
    """
    counts_per_revolution = 1024  
    wheel_circumference = 0.04  
    revolutions_per_second = counts_per_second / counts_per_revolution
    velocity = revolutions_per_second * wheel_circumference
    return velocity


def state_tuple_to_tensor(state, batch=True, dtype=torch.float32, device=None):
    """
    Convert a state tuple/array into a torch tensor suitable for the controller.

    Input state format (expected):
      (x_counts, theta_rad, v_counts_per_s, omega_rad_per_s)

    Output tensor format (float32):
      [x_m, theta_rad, v_m_s, omega_rad_s]

    Parameters
    - state: sequence or array-like of length 4 (or a 1D numpy array).
    - batch: if True, returns shape (1,4); otherwise returns shape (4,).
    - dtype: torch dtype (default torch.float32).
    - device: optional torch device.

    Returns
    - torch.Tensor containing the converted state.
    """
    # state: tuple or array-like of shape (4,) returns torch tensor of shape (4,)
    arr = np.asarray(state)

    def convert_one(s):
        # s expected length 4: x_counts, theta, v_counts, omega
        x_counts = int(s[0])
        theta = float(s[1])
        v_counts = int(s[2])
        omega = float(s[3])

        x_m = counts_to_meters(x_counts)
        v_m_s = countpersecond_to_meterspersecond(v_counts)

        # final order expected by controller: x, theta, v, omega
        return np.array([x_m, theta, v_m_s, omega], dtype=np.float32)

    
    conv = convert_one(arr)
    t = torch.as_tensor(conv, dtype=dtype, device=device)
    if batch:
        t = t.unsqueeze(0)
    return t


def u_converted(t) -> float:
    """
    Convert a control tensor or array-like to a python float.

    Parameters
    - t: torch.Tensor or array-like representing control (batched or unbatched).

    Returns
    - single float value (first element / first batch).
    """
    # not used in main skript
    # takes tensor or array-like, returns single float
    return float(t.detach().cpu().numpy().squeeze().item())



# SACZOP specific helpers below
def compute_reward(state, force) -> float:
    """
    Computes the reward for a hardware state, using the reward function named by
    HARDWARE_REWARD in simulation_zaczop/rewards.py (currently "cos_bonus_spin").
    Change HARDWARE_REWARD to switch; "default" is the original hand-written formula.

    Args:
        state: The state AFTER the step, as the Arduino reports it:
            (x [counts], theta [rad], v [counts/s], thetadot [rad/s], tripped).
        force: The commanded force [N].
    Returns:
        A float representing the computed reward.
    """

    x, theta, v, thetadot, tripped = state

    # the registry works in SI units; hardware frames carry x in encoder counts
    # and v in counts/s
    state_si = (
        counts_to_meters(x),
        theta,
        countpersecond_to_meterspersecond(v),
        thetadot,
        tripped,
    )
    return float(_reward_fn(state_si, force))


def reward_eval(x):
    """Compute reward based on position only.
    
    Args:
        x: position in counts
    
    Returns:
        reward: scaled to give ~20-30 total reward over 15 seconds
    """
    x_m = abs(counts_to_meters(x))
    # Scale down by factor of ~15-20 to get reasonable accumulated values
    return x_m 


def done_eval(state: tuple, current_step: int, max_ep_steps: int, x_threshold: float = X_TERM_M) -> bool:
    """
    Evaluates whether the episode should terminate based on the current state.
    Args:
        state: A tuple containing the current state (x, theta, v, thetadot, tripped).
        current_step: The current step count in the episode.
        max_ep_steps: The maximum allowed steps per episode.
        x_threshold: The position threshold for termination.
    Returns:
        A boolean indicating whether the episode is done.
    """
    
    x, theta, v, thetadot, tripped = state

    if tripped:               # Hardware-Safety
        return True
    if abs(counts_to_meters(x)) > x_threshold:  # Wagen zu weit
        return True
    if current_step > max_ep_steps:  # Max Steps erreicht
        return True

    return False



def sac_state_to_tensor(state: tuple, batch=True, dtype=torch.float32, device=None):
    """
    Convert a state tuple/array into a torch tensor suitable for the controller.
    Input state format (expected):
      (x_counts, theta_rad, v_counts_per_s, omega_rad_per_s, tripped)
    Output tensor format (float32):
      [x_m, theta_rad, v_m_s, omega_rad_s]
    """
    
    # state: tuple or array-like of shape (4,) returns torch tensor of shape (4,)
    arr = np.asarray(state)

    def convert_one(s):
        # s expected length >=4: x_counts, theta, v_counts, omega
        # If a fifth element `tripped` is present it will be ignored.
        if len(s) < 4:
            raise ValueError("state must have at least 4 elements")

        x_counts = int(s[0])
        theta = float(s[1])
        v_counts = int(s[2])
        omega = float(s[3])

        x_m = counts_to_meters(x_counts)
        v_m_s = countpersecond_to_meterspersecond(v_counts)

        # final order expected by controller: x, theta, v, omega
        return np.array([x_m, theta, v_m_s, omega], dtype=np.float32)

    
    conv = convert_one(arr)
    t = torch.as_tensor(conv, dtype=dtype, device=device)
    if batch:
        t = t.unsqueeze(0)
    return t



