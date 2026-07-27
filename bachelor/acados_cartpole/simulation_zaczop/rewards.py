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

    # # bonus for being upright and slow
    # if abs(theta) < 0.15 and abs(thetadot) < 1.5:
    #     reward += 0.5

    return float(max(reward, 0.0))




@register("cosine")
def compute_reward_cos(state, force) -> float:
    x, theta, v, thetadot, tripped = state
    x_max = 0.39

    reward  = np.cos(theta)                 # +1 oben (theta=0), -1 unten (theta=+-pi)
    reward -= 0.3 * (x / x_max) ** 2        # Positions-Strafe, glatt
    reward -= 0.001 * thetadot ** 2         # milder Spin-Penalty

    if abs(theta) < 0.15 and abs(thetadot) < 1.5:
        reward += 0.5
    return float(reward)


# === Quadratische Variante ===============================================
@register("quadratic")
def compute_reward_quad(state, force) -> float:
    x, theta, v, thetadot, tripped = state
   
    x_max = 0.39

    reward  = -1.0 * (theta / np.pi) ** 2   # 0 oben, -1 unten (theta=+-pi)
    reward -= 0.3 * (x / x_max) ** 2        # Positions-Strafe
    reward -= 0.001 * thetadot ** 2         # Spin-Penalty

    if abs(theta) < 0.15 and abs(thetadot) < 1.5:
        reward += 6

    return float(reward)


# === Strikt positive Varianten (kein suicidal-tripping) ==================
# terminated == tripped, daher: positive Rewards => "länger leben" lohnt sich
# => der Agent meidet die Wand von selbst, ohne Strafterm.

@register("cos_pos")
def compute_reward_cos_pos(state, force) -> float:
    """Strikt positiver, glatter Swingup-Reward in [0, 1].

    Keine negativen Terme, keine Cliffs. Position und Spin wirken nur als
    *multiplikative* Faktoren in [0.5, 1], dämpfen also sanft, ohne das
    Aufschwing-Signal je auf 0 zu ziehen.
    """
    x, theta, v, thetadot, tripped = state
    x_max = 0.39

    upright  = 0.5 * (1.0 + np.cos(theta))      # 1 oben, 0 unten, dicht & glatt
    centered = np.exp(-(x / x_max) ** 2)        # 1 in der Mitte -> ~0 am Rand
    calm     = np.exp(-(thetadot / 8.0) ** 2)   # 1 bei ruhigem Pendel

    reward = upright * (0.5 + 0.5 * centered) * (0.5 + 0.5 * calm)
    return float(reward)




#TODO: put into realsaczop but check that the state has counts and not meters
# so needs to be converted in real script for sim is ok!!
@register("cos_bonus")
def compute_reward_cos_bonus(state, force) -> float:
    """Dichtes positives Basissignal + glatte (gaußsche) Upright-Prämie, ~[0, 1.5].

    Ersetzt den harten `if abs(theta)<0.15: += const`-Sprung durch eine glatte
    Beule, die SACs Critic deutlich leichter fittet.
    """
    x, theta, v, thetadot, tripped = state
    x_max = 0.39

    upright  = 0.5 * (1.0 + np.cos(theta))                                   # [0,1] dicht
    balanced = np.exp(-(theta / 0.25) ** 2) * np.exp(-(thetadot / 2.0) ** 2) * 3 # [0,1] glatte Praemie
    reward = (upright + 0.5 * balanced) * np.exp(-0.5 * (x / x_max) ** 2)
    return float(reward)

@register("cos_bonus_only_theta")
def compute_reward_cos_bonus(state, force) -> float:
    """Dichtes positives Basissignal + glatte (gaußsche) Upright-Prämie, ~[0, 1.5].

    Ersetzt den harten `if abs(theta)<0.15: += const`-Sprung durch eine glatte
    Beule, die SACs Critic deutlich leichter fittet.
    """
    x, theta, v, thetadot, tripped = state
    x_max = 0.39

    upright  = 0.5 * (1.0 + np.cos(theta))                                   # [0,1] dicht
    balanced = np.exp(-(theta / 0.25) ** 2) * np.exp(-(thetadot / 2.0) ** 2) * 3 # [0,3] glatte Praemie
    reward = (upright +  balanced) 
    return float(reward)