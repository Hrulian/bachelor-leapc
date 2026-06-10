"""Reward functions for the simulated cartpole swingup.

All rewards operate on the SI state AFTER the step (same timing as
compute_reward in the real script):

    state = (x [m], theta [rad], v [m/s], thetadot [rad/s], tripped)
    force = commanded force [N]

theta convention: 0 = upright, +-pi = hanging down.

To test a new reward, add a function here and decorate it with
@register("my_name"), then run:

    python sim_sac_zop.py --reward my_name
"""

import numpy as np

REWARDS: dict = {}


def register(name: str):
    def deco(fn):
        REWARDS[name] = fn
        return fn
    return deco


def get_reward_fn(name: str):
    try:
        return REWARDS[name]
    except KeyError:
        raise KeyError(f"Unknown reward '{name}'. Available: {sorted(REWARDS)}") from None


@register("default")
def reward_default(state, force) -> float:
    """SI-unit port of compute_reward in my_helpers.py (the one used on hardware)."""
    x, theta, v, thetadot, tripped = state

    # swingup reward, max 0.1 when upright
    reward = abs(np.pi - abs(theta)) / (10.0 * np.pi)

    # position penalty: zero in center, -0.1 at the rail ends
    reward -= 0.1 * (abs(x) / 0.39)

    # discourage high angular velocities
    if abs(thetadot) > 12.0:
        reward = 0.0

    # bonus for being upright and slow
    if abs(theta) < 0.15 and abs(thetadot) < 1.5:
        reward += 0.5

    return float(max(reward, 0.0))


@register("cosine")
def reward_cosine(state, force) -> float:
    """Smooth cosine shaping: 1 upright, 0 hanging down, with small penalties."""
    x, theta, v, thetadot, tripped = state
    upright = 0.5 * (1.0 + np.cos(theta))           # in [0, 1]
    reward = 0.1 * upright
    reward -= 0.02 * (abs(x) / 0.39)
    reward -= 0.001 * abs(force) / 20.0
    if abs(theta) < 0.15 and abs(thetadot) < 1.5:
        reward += 0.5
    return float(max(reward, 0.0))


@register("energy")
def reward_energy(state, force) -> float:
    """Energy-based shaping for swingup + upright bonus for stabilization.

    Rewards approaching the energy of the upright equilibrium, which often
    speeds up learning of the swingup phase.
    """
    x, theta, v, thetadot, tripped = state
    m, l, g = 0.016, 0.18, 9.81
    # pendulum energy relative to upright (theta=0): E_target = 0
    e_kin = 0.5 * m * (l * thetadot) ** 2
    e_pot = m * g * l * (np.cos(theta) - 1.0)       # 0 upright, -2mgl down
    e_err = abs(e_kin + e_pot)                      # 0 when on the homoclinic orbit
    reward = 0.1 * np.exp(-10.0 * e_err)
    reward -= 0.02 * (abs(x) / 0.39)
    if abs(theta) < 0.15 and abs(thetadot) < 1.5:
        reward += 0.5
    return float(max(reward, 0.0))
