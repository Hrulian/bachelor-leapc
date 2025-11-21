import numpy as np
import time

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
    state = np.array([[0.22, -0.35, 0.15, 0.21]], dtype=float)  # Shape: (1, 4)

    # reset history/set it to None
    ctx = None

    # pass the controller the current state and measure time
    t0 = time.perf_counter()
    ctx, u = controller(state, ctx)
    dt_ms = (time.perf_counter() - t0) * 1e3

    # print u for testing
    print(u)
    print(f"controller() took {dt_ms:.2f} ms")



if __name__ == '__main__':
    main()

















# test state und control as delivered from the Arduino





