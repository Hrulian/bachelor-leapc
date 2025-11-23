import numpy as np
import time
import torch
import matplotlib.pyplot as plt  # added for plotting
from helpers import state_tuple_to_tensor, u_converted

from my_parameter_manager import create_custom_cartpole_params
from leap_c.examples.cartpole.planner import CartPolePlannerConfig, CartPolePlanner


def main():
    # create configuration object
    cfg = CartPolePlannerConfig()

    # set the specific parameters 
    params = create_custom_cartpole_params()

    # create controller object
    controller = CartPolePlanner(cfg, params)

    # define test state: state = [x, v, xdot, vdot]. Later updated in main control loop
    state = (0.22, -0.35, 0.15, 0.21)  

    # reset history/set it to None
    ctx = None

    # prepare state as torch tensor for the planner
    state_converted = state_tuple_to_tensor(state)
        
    # call planner: returns (ctx, u0, x_traj, u_traj, value)
    t0 = time.perf_counter()
    ctx, u0, x_traj, u_traj, value = controller(state_converted, ctx=ctx)
    dt_ms = (time.perf_counter() - t0) * 1e3



    # for testing
    print("ctx type:", type(ctx))
    print("u0 type:", type(u0), "shape:", getattr(u0, "shape", None))
    print("u0 (numpy):", u_converted(u0))
    print("x_traj shape:", getattr(x_traj, "shape", None))
    print("u_traj shape:", getattr(u_traj, "shape", None))
    print(f"controller() took {dt_ms:.2f} ms")





if __name__ == "__main__":
    main()


