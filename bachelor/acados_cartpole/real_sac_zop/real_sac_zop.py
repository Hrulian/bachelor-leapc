"""
Plan structure for real_sac_zop.py_

-MPC layer
    - as before: ocp->planner->controllerfromplanner
    
-policy in parameter space
    -use MpcSacActor from sac_zop.py
    -input: state + context file 
    -output: param vector for MPC + actual control(force) + updated context for next run
    
-parameter critic + target copy
    - use two SacCritic from sac.py
    - create target critic which is periodically updated from utils.py soft_target_update
    
-ReplayBuffer
    - use existing ReplayBuffer from buffer.py
    -store (state, param, reward, next_state, done)
    - is filled constantly from real environment interaction in my main loop
    

---------------------------------------------------------------


Main Loop Pseudocode:
    -we have previous state s, paramter p used to gnerate action a
    -apply action a to real system, get next state s', compute rewards, done(?)
    -put in buffer
    -from new state s', get new parameter p' and action a' from MpcSacActor
    -periodically call the sac_zop update (prob put in a function) 
    -if done = true ask arduino for physical reset, reset ctx and start new episode
    
----------------------------------------------------------------


SAC-ZOP update step
- Reuse the idea from SacZopTrainer.train_loop (sac_zop.py):
    - Sample batch from ReplayBuffer: (o, a, r, o_prime, te)
    where a = param (the parameters used to generate the last action)
    - Policy on o, o_prime in parameter space (only_param=True)
    - Temperature alpha update (optional)
    - Critic update:
    - target = r + gamma * (1 - done) * [min Q_target(o_prime, param') - alpha * log pi(param'|o_prime)]
    - critic loss = MSE(Q(o,a), target)
    - Actor update:
    - actor loss = mean(alpha * log pi(param|o) - min Q(o, param))
    - Soft-update target critic with tau
- Implement this as a standalone function:
    sac_zop_update_step()
    that uses global (or passed-in) actor, critic, target_critic, log_alpha,
    optimizers, and replay buffer
"""
import torch 

from bachelor.acados_cartpole.real_sac_zop.my_sac import SacTrainerConfig, SacCritic
from bachelor.acados_cartpole.real_sac_zop.my_sac_zop import  MpcSacActor
from bachelor.acados_cartpole.real_sac_zop.my_utils import soft_target_update
from bachelor.acados_cartpole.real_sac_zop.my_buffer import ReplayBuffer
from bachelor.acados_cartpole.my_planner import CartPolePlannerConfig, CartPolePlanner, create_custom_cartpole_params
from bachelor.acados_cartpole.my_helpers import (
    force_to_pwm,
    state_tuple_to_tensor,
    counts_to_meters,
    countpersecond_to_meterspersecond,
)
from leap_c.planner import ControllerFromPlanner
from leap_c.torch.nn.mlp import MlpConfig
from leap_c.torch.nn.extractor import get_extractor_cls  # optional helper

# MPC Layer Setup
cfg_planner = CartPolePlannerConfig()
params = create_custom_cartpole_params("stagewise", cfg_planner.N_horizon)
planner = CartPolePlanner(cfg_planner, params)
controller_wrapped = ControllerFromPlanner(planner)


# SacZop config
# TODO: Define Hyperparameters as in sac_zop_run.py (just copy them over:D)
# i also think that in the config file is everything set and in the inits below you can refer to it
cfg_saczop = SacTrainerConfig()


# Replay Buffer init
replay_buffer = ReplayBuffer()


# critic init
critic = SacCritic()
target_critic = SacCritic()


# actor init
actor = MpcSacActor()


# entropy temperature ALpha init
log_alpha = torch.nn.Parameter()


# optimizers (stochastic gradient descent algorithm to opt my critics and actor networks)
critic_optimizer = torch.optim.Adam()
actor_optimizer = torch.optim.Adam()


