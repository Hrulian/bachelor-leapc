import numpy as np
import torch


def force_to_pwm(force: float, velocity: float) -> int:
    a = 40.24
    b = 6.95
    c = -0.21
    mass = 0.15  # [kg] estimated: 0.54 g cart + 2* 30g kuggellager + 24g * welle + 0.012 * unknown parts

    u = ((force / mass) + a * velocity + c * sgn(velocity)) / b

    # map u in [-24, 24] to pwm in [-255, 255]
    max_u = 24.0
    max_pwm = 255
    scaled = u * (max_pwm / max_u)

    # round and clamp to [-255, 255]
    pwm = int(round(scaled))
    pwm = max(-max_pwm, min(max_pwm, pwm))
    return pwm


def sgn(x: float) -> int:
    # Sign function that returns -1 for negative, 1 for positive, and 0 for zero
    return (x > 0) - (x < 0)


def counts_to_meters(counts: int) -> float:
    # Convert encoder counts to position in meters
    counts_per_revolution = 1024  
    wheel_circumference = 0.04  
    revolutions = counts / counts_per_revolution
    position = revolutions * wheel_circumference
    return position


def countpersecond_to_meterspersecond(counts_per_second: int) -> float:
    # Convert encoder counts per second to velocity in meters per second
    counts_per_revolution = 1024  
    wheel_circumference = 0.04  
    revolutions_per_second = counts_per_second / counts_per_revolution
    velocity = revolutions_per_second * wheel_circumference
    return velocity


def state_tuple_to_tensor(state, batch=True, dtype=torch.float32, device=None):
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
    # not used in main skript
    # takes tensor or array-like, returns single float
    return float(t.detach().cpu().numpy().squeeze().item())


