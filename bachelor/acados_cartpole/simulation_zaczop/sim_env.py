"""Simulation environment that mimics the real cartpole hardware setup.

The state convention matches what the host receives from the Arduino
(after the count->SI conversion done in real_sac_zop.py):

    [x (m), theta (rad), v (m/s), thetadot (rad/s)]

- theta is wrapped to [-pi, pi]; 0 = upright, +-pi = hanging down
  (Arduino sends -wrap_to_pi(theta)).
- thetadot is clipped to +-20 rad/s (same clip as in real_sac_zop.py).
- Termination when |x| > x_threshold = 0.39 m (same as done_eval / hardware trip).
- dt = 10 ms (control frequency of the real loop).
- Reset puts the pole straight down at the center, like a normal gym env.

The action is the force on the cart in Newtons, applied directly
(no force->PWM mapping).
"""

from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(kw_only=True)
class RealCartPoleSimConfig:
    """Configuration for the hardware-matched CartPole simulation."""

    # physics (identified parameters of the real system)
    gravity: float = 9.81
    masscart: float = 0.17444
    masspole: float = 0.016
    length: float = 0.18

    # timing / bounds (matching real_sac_zop.py)
    dt: float = 0.01             # control frequency of the real loop [s]
    x_threshold: float = 0.39    # CartPolePlannerConfig.x_threshold
    Fmax: float = 20.0           # CartPolePlannerConfig.Fmax
    thetadot_limit: float = 20.0  # clip in real_sac_zop.py


def wrap_to_pi(theta: float) -> float:
    """Wrap an angle to [-pi, pi], same convention as the Arduino firmware."""
    return ((theta + np.pi) % (2.0 * np.pi)) - np.pi


class RealCartPoleSimEnv(gym.Env):
    """CartPole swingup environment matched to the real hardware setup.

    The reward is computed by an injected callable so that different reward
    functions can be tested without touching the environment:

        reward_fn(state, force) -> float

    where state = (x [m], theta [rad], v [m/s], thetadot [rad/s], tripped)
    is the state AFTER the step (same as compute_reward in the real script,
    which is called on the new state) and force is the commanded force [N].
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_fn: Callable[[tuple, float], float],
        cfg: RealCartPoleSimConfig | None = None,
        max_episode_steps: int = 1000,
    ):
        self.cfg = RealCartPoleSimConfig() if cfg is None else cfg
        self.reward_fn = reward_fn
        self.max_episode_steps = max_episode_steps

        # identical to the obs space defined in real_sac_zop.py
        obs_low = np.array([-self.cfg.x_threshold, -np.pi, -5, -21], dtype=np.float32)
        obs_high = np.array([self.cfg.x_threshold, np.pi, 5, 21], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(-self.cfg.Fmax, self.cfg.Fmax, dtype=np.float32)

        self.x: np.ndarray | None = None
        self.step_count = 0
        self.reset_needed = True

    # dynamics ---------------------------------------------------------------
    def _f_explicit(self, x: np.ndarray, force: float) -> np.ndarray:
        g = self.cfg.gravity
        M = self.cfg.masscart
        m = self.cfg.masspole
        l = self.cfg.length  # noqa: E741
        _, theta, dx, dtheta = x
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        denominator = M + m - m * cos_theta * cos_theta
        return np.array(
            [
                dx,
                dtheta,
                (-m * l * sin_theta * dtheta * dtheta + m * g * cos_theta * sin_theta + force)
                / denominator,
                (
                    -m * l * cos_theta * sin_theta * dtheta * dtheta
                    + force * cos_theta
                    + (M + m) * g * sin_theta
                )
                / (l * denominator),
            ]
        )

    def _rk4_step(self, x: np.ndarray, force: float, h: float) -> np.ndarray:
        k1 = self._f_explicit(x, force)
        k2 = self._f_explicit(x + 0.5 * h * k1, force)
        k3 = self._f_explicit(x + 0.5 * h * k2, force)
        k4 = self._f_explicit(x + h * k3, force)
        return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # gym API ----------------------------------------------------------------
    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.reset_needed:
            raise Exception("Call reset before using the step method.")

        force = float(np.clip(np.asarray(action).squeeze(), -self.cfg.Fmax, self.cfg.Fmax))

        self.x = self._rk4_step(self.x, force, self.cfg.dt)
        self.x[1] = wrap_to_pi(self.x[1])
        self.x[3] = np.clip(self.x[3], -self.cfg.thetadot_limit, self.cfg.thetadot_limit)
        self.step_count += 1

        tripped = bool(abs(self.x[0]) > self.cfg.x_threshold)
        terminated = tripped
        truncated = self.step_count > self.max_episode_steps

        state = (float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3]), int(tripped))
        reward = float(self.reward_fn(state, force))

        self.reset_needed = terminated or truncated
        info = {"tripped": tripped}
        return self.x.astype(np.float32), reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # pole hanging down at the center, everything still
        self.x = np.array([0.0, np.pi, 0.0, 0.0], dtype=np.float64)
        self.step_count = 0
        self.reset_needed = False
        return self.x.astype(np.float32), {}
