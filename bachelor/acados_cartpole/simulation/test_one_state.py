#!/usr/bin/env python3
"""
Minimal test script: initialize the CartPole planner, pass one state,
and print the planner force output `u_force` and the mapped PWM `u`.

Usage examples:
  python3 test_one_state.py                    # uses defaults
  python3 test_one_state.py --x 0.0 --theta 0.1 --v 0.0 --thetadot 0.0
"""

import sys

try:
    from my_helpers import state_tuple_to_tensor, countpersecond_to_meterspersecond, force_to_pwm
    from my_planner import CartPolePlannerConfig, CartPolePlanner, create_custom_cartpole_params
except Exception:
    print("Failed to import project modules. Run this from the `bachelor/acados_cartpole` package root.")
    raise


# Define the single test state here as (x, theta, v, thetadot)
# x: counts, theta: rad, v: counts/s, thetadot: rad/s
STATE = (0.0, 0.0, 0.0, 0.0)


def main():
    # create planner (same minimal init as in MPC_REAL.py)
    cfg = CartPolePlannerConfig()
    params = create_custom_cartpole_params("stagewise", cfg.N_horizon)
    controller = CartPolePlanner(cfg, params)

    # use the predefined STATE tuple
    state_tuple = STATE

    # convert to model/tensor input
    state_tensor = state_tuple_to_tensor(state_tuple)

    # call the controller once
    ctx = None
    try:
        ctx, u0, x_traj, u_traj, value = controller(state_tensor, ctx=ctx)
    except Exception as e:
        print("Controller call failed:", e)
        sys.exit(1)

    # extract scalar force from returned u0 (torch tensor)
    try:
        u_force = float(u0.detach().cpu().numpy().squeeze().item())
    except Exception:
        # fallback if tensor shape differs
        u_force = float(u0)

    # map force -> pwm using the helper (v must be converted to m/s first)
    v_counts_per_s = state_tuple[2]
    u_pwm = force_to_pwm(u_force, countpersecond_to_meterspersecond(v_counts_per_s))

    print(f"Input state: x={state_tuple[0]}, theta={state_tuple[1]}, v={state_tuple[2]}, thetadot={state_tuple[3]}")
    print(f"Planner force output (N): {u_force:.6f}")
    print(f"Mapped PWM (after force_to_pwm): {u_pwm:.6f}")


if __name__ == '__main__':
    main()
