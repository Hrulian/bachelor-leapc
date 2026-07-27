"""MPC simulation for the cart-pole system using acados and a custom planner.

Closed-loop test of the pure MPC controller on the SAME hardware-matched
simulation used for the reward tests (simulation_zaczop/sim_env.py):

    dt = 0.01 s, x_threshold = 0.39 m, theta wrapped to [-pi, pi],
    thetadot clipped to +-20 rad/s, pole starts hanging down at the center.

Records one rollout and shows the trajectory (x, theta, dx, dtheta, u).
"""

from argparse import ArgumentParser
from collections import Counter

import numpy as np
import torch

from bachelor.acados_cartpole.my_helpers import u_converted
from bachelor.acados_cartpole.simulation_zaczop.planner_registry import (
    build_planner,
    PLANNER_REGISTRY,
)
from bachelor.acados_cartpole.simulation_zaczop.sim_env import (
    RealCartPoleSimEnv,
    RealCartPoleSimConfig,
)
from bachelor.acados_cartpole.my_utils_plot import plot_pendulum


# 20 s rollout at dt=0.01 s -> 2000 steps (env truncates at max_episode_steps)
MAX_EP_STEPS = 200


def main():
    p = ArgumentParser(description="Closed-loop MPC simulation")
    p.add_argument("--planner", type=str, default="full", choices=sorted(PLANNER_REGISTRY),
                   help="Which planner/OCP to use (see planner_registry.py)")
    args = p.parse_args()

    # hardware-matched sim, identical to the reward-test environment
    env_cfg = RealCartPoleSimConfig()
    env = RealCartPoleSimEnv(
        reward_fn=lambda state, force: 0.0,  # MPC test ignores the reward
        cfg=env_cfg,
        max_episode_steps=MAX_EP_STEPS,
    )

    # planner / OCP selected from the registry; the env always passes the full
    # [x, theta, v, thetadot] state, so each planner adapts internally
    controller = build_planner(args.planner)
    print(f"Planner: {args.planner}")

    obs, _ = env.reset()
    ctx = None

    # record one closed-loop rollout for the trajectory plot
    states  = [np.asarray(obs, dtype=np.float32)]  # X_true: (N+1, 4) -> [x, theta, dx, dtheta]
    actions = []                                   # U:      (N, 1)   -> applied force [N]

    print("Running MPC rollout ...")

    # QP/NLP solver health tracking: count how many controller calls ended with a
    # non-zero acados status (QP failure, max iter, NaN, ...) or raised.
    n_calls = 0
    n_solver_fail = 0
    status_counts = Counter()

    while True:
        state_t = torch.as_tensor(np.array(obs, dtype=np.float32)).unsqueeze(0)

        try:
            ctx, u0, _, _, value = controller(state_t, ctx=ctx)
        except Exception:
            u0 = None
            ctx = None

        # record solver status for this call (-1 = exception / no solution)
        n_calls += 1
        status_code = -1 if ctx is None else int(np.asarray(ctx.status).flat[0])
        status_counts[status_code] += 1
        if status_code != 0:
            n_solver_fail += 1

        if u0 is None:
            action = 0.0
        else:
            action = u_converted(u0)
            if not np.isfinite(action) or abs(float(action)) > 1e6:
                action = 0.0
                ctx = None
        action = float(np.clip(action, -env_cfg.Fmax, env_cfg.Fmax))

        obs, _, terminated, truncated, _ = env.step(np.array([action], dtype=np.float32))

        actions.append(action)
        states.append(np.asarray(obs, dtype=np.float32))

        if terminated or truncated:
            break

    env.close()

    X = np.array(states)                       # (N+1, 4)
    U = np.array(actions).reshape(-1, 1)       # (N, 1)
    t = np.arange(X.shape[0]) * env_cfg.dt     # (N+1,)

    reason = "tripped (|x| > x_threshold)" if terminated else "time limit"
    print(f"Rollout finished: {len(actions)} steps, {t[-1]:.2f} s simulated ({reason})")

    # solver error rate over the whole rollout
    fail_pct = 100.0 * n_solver_fail / max(n_calls, 1)
    code_meaning = {
        0: "success",
        1: "max NLP/QP iter",
        2: "min step size",
        3: "NaN detected",
        4: "QP solver failure",
        -1: "exception / no solution",
    }
    print(f"QP solver errors: {n_solver_fail}/{n_calls} calls failed ({fail_pct:.2f}%)")
    for code in sorted(status_counts):
        print(f"  status {code:>2} ({code_meaning.get(code, 'unknown')}): {status_counts[code]}")

    plot_pendulum(
        t, env_cfg.Fmax, U, X,
        x_labels=['$x$ [m]', r'$\theta$ [rad]', r'$\dot{x}$ [m/s]', r'$\dot{\theta}$ [rad/s]'],
        u_labels=['$u$ [N]'],
    )


if __name__ == "__main__":
    main()
