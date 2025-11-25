import os

import numpy as np
from acados_template import AcadosOcpSolver
from gymnasium.wrappers import RecordVideo
from setup_ocp import setup_ocp
from utils import plot_pendulum

from leap_c.examples.cartpole.env import CartPoleEnv, CartPoleEnvConfig


def main():
    Fmax = 10

    Tf = 1.
    N_horizon = 20
    
    ocp = setup_ocp(Fmax, N_horizon, Tf)
    ocp_solver = AcadosOcpSolver(ocp)

    cfg_env = CartPoleEnvConfig(max_time=5.0)
    # TODO Exercise 1.1: Read the docstring of this class to 
    # make yourself familiar with the environment.
    env = CartPoleEnv(render_mode="rgb_array", cfg=cfg_env)
    
    # Wrap environment for video recording
    env = RecordVideo(
        env, os.getcwd(), name_prefix="cartpole_video", episode_trigger=lambda x: True
    )

    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu
 
    # The number of steps taken until truncation
    Nsim = int(cfg_env.max_time / cfg_env.dt) + 1
    
    simX = np.zeros((Nsim+1, nx))
    simU = np.zeros((Nsim, nu))

    # Start the episode by calling env.reset()
    simX[0,:], _ = env.reset(seed=1337)

    t = np.zeros((Nsim))

    # closed loop
    i = 0
    trunc = False
    term = False
    cum_reward = 0.0
    while not (trunc or term):

        # solve ocp and get next control input
        simU[i,:] = ocp_solver.solve_for_x0(x0_bar = simX[i, :], fail_on_nonzero_status=False)
        
        # Only for rendering
        iterate = ocp_solver.store_iterate_to_obj()
        state_trajectory = iterate.x_traj
        env.unwrapped.include_this_state_trajectory_to_rendering(state_trajectory)

        t[i] = ocp_solver.get_stats('time_tot')

        # simulate system / Make a step in the environment
        x_prime, r, term, trunc, info = env.step(action=simU[i,:])
        simX[i+1, :] = x_prime
        cum_reward += r

        i += 1

    # evaluate timings and print reward
    # scale to milliseconds
    t *= 1000
    print(f'Computation time in ms: min {np.min(t):.3f} median ' 
          f'{np.median(t):.3f} max {np.max(t):.3f}')
    print(f'Achieved cumulative reward: {cum_reward:.3f}')
    if info["task"]["success"]:
        print('Successfully balanced the pole!')
    else:
        print('Failed to balance the pole.')

    # plot results
    model = ocp_solver.acados_ocp.model
    plot_pendulum(np.linspace(0, (Tf/N_horizon)*Nsim, Nsim+1), Fmax, simU, simX, latexify=False, 
                  time_label=model.t_label, x_labels=model.x_labels, u_labels=model.u_labels)
    
    env.close()


if __name__ == '__main__':
    main()
