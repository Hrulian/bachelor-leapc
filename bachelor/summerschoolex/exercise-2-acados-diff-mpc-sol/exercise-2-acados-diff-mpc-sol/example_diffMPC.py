# -*- coding: future_fstrings -*-
#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#
import numpy as np
import torch
from setup_ocp_with_manager import create_cartpole_params, setup_ocp_with_manager
from utils import plot_sol_and_sens

from leap_c.ocp.acados.controller import AcadosController
from leap_c.ocp.acados.parameters import AcadosParameterManager
from leap_c.ocp.acados.torch import AcadosDiffMpcTorch


def main():
    N_horizon = 5
    T_horizon = 0.25
    Fmax = 80.0
    delta_p = 0.01
    
    p_values_npy = np.arange(0.5*np.pi, 1.5*np.pi, delta_p)
    p_values_npy = p_values_npy.reshape(-1, 1) # Shape (N, 1)
    p_values = torch.tensor(p_values_npy, dtype=torch.float32)
    # Enable gradient computation for this tensor
    p_values.requires_grad = True
    
    # TODO Exercise 2.4: Create the controller
    # NOTE: Set n_batch_max high in AcadosDiffMpc enough to cover the batch size of p_values
    params = create_cartpole_params()
    param_manager = AcadosParameterManager(params, N_horizon)
    ocp = setup_ocp_with_manager(Fmax, N_horizon, T_horizon, param_manager)
    diff_mpc = AcadosDiffMpcTorch(ocp, n_batch_max=p_values.shape[0])
    controller = AcadosController(param_manager, diff_mpc)
    
    x0_npy = np.array([0.0, np.pi, 0.0, 0.0])
    x0_npy = np.tile(x0_npy, (p_values_npy.shape[0], 1)) # Shape (N, 4)
    x0 = torch.tensor(x0_npy, dtype=torch.float32)
    
    # Solve the batch of OCPs
    ctx, u0 = controller.forward(obs=x0, param=p_values)
    assert not np.count_nonzero(ctx.status) > 0, "Solver failed for some samples."

    # Obtain the policy gradient du0/dp
    du0_dp = controller.jacobian_action_param(ctx).flatten()
    
    u0_npy = u0.detach().numpy().flatten()

    du0_dp_fd = np.gradient(u0_npy, delta_p)
    
    test_tol = 1e-3
    median_diff = np.median(np.abs(du0_dp - du0_dp_fd))
    print("Median difference between policy gradient obtained by acados and via FD is "
          f"{median_diff} should be < {test_tol}.")
    
    # solutions to plot
    label = r'solver'
    pi_label_pairs = []
    sens_pi_label_pairs = []

    pi_label_pairs.append(label)
    sens_pi_label_pairs.append(label)

    sens_pi_label_pairs.append('finite diff.')

    plot_sol_and_sens(x_values=p_values.detach().numpy(), pis = [u0_npy], 
                      senss = [du0_dp, du0_dp_fd], pi_labels=pi_label_pairs, 
                      sens_labels=sens_pi_label_pairs)


if __name__ == "__main__":
    main()
