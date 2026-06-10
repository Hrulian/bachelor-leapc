"""MPC simulation for cart-pole system using acados and a custom planner
This script sets up a closed-loop simulation of a cart-pole system using an MPC controller.
important note: Real system cant exceed certain limits on position and velocity
x treshold must be ALWAYS < 0.45 better if 0.4 (way better)
velocity should be below 2 m/s (otherwise real system fails) -> set Fmax accordingly in planner
friction is not modeled in env nor the planner currently.
Also the slack variables on the x_treshold constraints should be kept in the planner to avoid infeasible QPs
-> but they are inside now anyway since otherwise one is getting lot of QP errors
-> Sim works by either incresing the x treshhold or the Fmax in the planner
-> but both lead to an unrealistic simulated scenario
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

from bachelor.acados_cartpole.my_acados_ocp import create_custom_cartpole_params
from bachelor.acados_cartpole.my_helpers import u_converted
from bachelor.acados_cartpole.my_planner import CartPolePlannerConfig, CartPolePlanner
from bachelor.acados_cartpole.my_env import CartPoleEnv, CartPoleEnvConfig, CartPoleBalanceEnv
from bachelor.acados_cartpole.my_utils_plot import plot_pendulum


N_WARMUP = 20   # calls ignored for timing (solver JIT / cache warmup)
N_BENCH  = 200  # calls to measure


def main():
    env_cfg = CartPoleEnvConfig(max_time=20.0)
    env = CartPoleBalanceEnv(render_mode=None, cfg=env_cfg)

    cfg = CartPolePlannerConfig()
    params = create_custom_cartpole_params("stagewise", cfg.N_horizon)
    controller = CartPolePlanner(cfg, params)

    obs, _ = env.reset()
    ctx = None

    times_ms = []
    total_calls = 0

    print(f"Warmup ({N_WARMUP} calls) + Benchmark ({N_BENCH} calls) ...")

    while len(times_ms) < N_BENCH:
        state_t = torch.as_tensor(np.array(obs, dtype=np.float32)).unsqueeze(0)

        t0 = time.perf_counter()
        try:
            ctx, u0, _, _, value = controller(state_t, ctx=ctx)
        except Exception:
            u0 = None
            ctx = None
            value = np.array([np.inf])
        dt_ms = (time.perf_counter() - t0) * 1e3
        total_calls += 1

        if total_calls > N_WARMUP:
            times_ms.append(dt_ms)

        if u0 is None:
            action = 0.0
        else:
            action = u_converted(u0)
            if not np.isfinite(action) or abs(float(action)) > 1e6:
                action = 0.0
                ctx = None
        action = float(np.clip(action, -env.unwrapped.cfg.Fmax, env.unwrapped.cfg.Fmax))

        obs, _, terminated, truncated, _ = env.step(np.array([float(action)], dtype=np.float32))
        if terminated or truncated:
            obs, _ = env.reset()
            ctx = None

    env.close()

    t = np.array(times_ms)
    print(f"\n── MPC call timing ({N_BENCH} samples, after {N_WARMUP} warmup calls) ──")
    print(f"  median  {np.median(t):.2f} ms")
    print(f"  mean    {np.mean(t):.2f} ms")
    print(f"  p10     {np.percentile(t, 10):.2f} ms")
    print(f"  p90     {np.percentile(t, 90):.2f} ms")
    print(f"  p99     {np.percentile(t, 99):.2f} ms")
    print(f"  max     {np.max(t):.2f} ms")


if __name__ == "__main__":
    main()


