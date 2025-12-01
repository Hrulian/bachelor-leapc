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

from my_acados_ocp import create_custom_cartpole_params
from my_helpers import u_converted
from my_planner import CartPolePlannerConfig, CartPolePlanner
from my_env import CartPoleEnv, CartPoleEnvConfig, CartPoleBalanceEnv
from utils import plot_pendulum


def main():
    # environment (use rgb_array so RecordVideo can capture frames)
    env_cfg = CartPoleEnvConfig(max_time=20.0)
    # use the balance environment so the pole starts upright
    env = CartPoleBalanceEnv(render_mode="rgb_array", cfg=env_cfg)

    # planner setup - align planner bounds with the env to avoid infeasible QPs
    cfg = CartPolePlannerConfig()
    params = create_custom_cartpole_params("stagewise", cfg.N_horizon)
    controller = CartPolePlanner(cfg, params)
    ctx = None

    # record video to the same folder as this script
    video_folder = Path(__file__).resolve().parent
    env = RecordVideo(env, str(video_folder), name_prefix="simulation_video", episode_trigger=lambda _: True)

    obs, _ = env.reset()  # obs layout (env): [x, theta, dx, dtheta]

    # data logging for plotting (we will use plot_pendulum)
    X_traj = []  # sequence of observations (N_sim, nx)
    U_traj = []  # sequence of actions (N_sim-1, nu)

    # push initial observation
    X_traj.append(np.array(obs, dtype=np.float32))

    # main closed-loop
    max_steps = 200
    for step in range(max_steps):
        # env observation -> planner state ordering: [x, theta, dx, dtheta]
        x_env, theta_env, dx_env, dtheta_env = obs
        state_arr = np.array([x_env, theta_env, dx_env, dtheta_env], dtype=np.float32)
        # prepare torch tensor (batch dim)
        state_t = torch.as_tensor(state_arr).unsqueeze(0)

        # call planner (warm-start ctx)
        t0 = time.perf_counter()
        try:
            ctx, u0, x_traj, u_traj, value = controller(state_t, ctx=ctx)
        except Exception as e:
            print(f"Planner raised exception: {e}; using safe fallback u=0 and clearing ctx")
            u0 = None
            ctx = None
            value = np.array([np.inf])
        dt_ms = (time.perf_counter() - t0) * 1e3

        # extract first control action (float, in N)
        if u0 is None:
            action = 0.0
        else:
            # convert and guard against NaN / huge values
            action = u_converted(u0)
            if not np.isfinite(action) or abs(float(action)) > 1e6 or float(value[0]) > 1e6:
                print("Warning: bad solver output (NaN/inf/huge). Falling back to safe action=0 and resetting warm-start.")
                action = 0.0
                ctx = None

        # clip to environment limits as a final safety
        action = float(np.clip(action, -env.unwrapped.cfg.Fmax, env.unwrapped.cfg.Fmax))

        # log action (shape (1,))
        U_traj.append(np.array([float(action)], dtype=np.float32))

        # apply action to env
        obs, reward, terminated, truncated, info = env.step(np.array([float(action)], dtype=np.float32))
        done = bool(terminated or truncated)

        # RecordVideo will call render() internally; still safe to call
        try:
            env.render()
        except Exception:
            pass

        print(f"step {step:03d} dt={dt_ms:.2f}ms action={float(action):.3f} cost={float(value[0]):.3f}")

        # append new observation
        X_traj.append(np.array(obs, dtype=np.float32))

        if done:
            break

    # close env (this finalizes video file)
    try:
        env.close()
    except Exception:
        pass

    # convert to arrays for plotting: X shape (N_sim, nx), U shape (N_sim-1, nu)
    X_arr = np.vstack(X_traj)
    U_arr = np.vstack(U_traj) if len(U_traj) > 0 else np.zeros((0, 1), dtype=np.float32)

    # time vector (one entry per state)
    t_vec = np.arange(X_arr.shape[0]) * env.unwrapped.cfg.dt

    # plot using provided utility and save figure next to the script
    plot_pendulum(t_vec, env.unwrapped.cfg.Fmax, U_arr, X_arr, latexify=False, plt_show=False)
    fig_path = video_folder / "simulation_trajectory.png"
    plt.savefig(fig_path)
    print(f"Saved trajectory plot to {fig_path}")

    print(f"Recorded video(s) saved to: {video_folder}")

    # optionally display the plot
    plt.show()


if __name__ == "__main__":
    main()


