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
        -done: cart to far, pole to tilted(if onyl balancing), max steps reached
        
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
    that uses global -,actor, -,critic -,target_critic -,log_alpha,
    optimizers, and replay buffer
"""
import torch 
import numpy as np
import gymnasium as gym
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

#initialize global variables for sac_zop update step
global step_count
step_count = 0


#LEARNING INIT###############################################################################
# device setup
device = "cpu"


# MPC Layer Setup
cfg_planner = CartPolePlannerConfig()
params = create_custom_cartpole_params("stagewise", cfg_planner.N_horizon)
planner = CartPolePlanner(cfg_planner, params)
controller_wrapped = ControllerFromPlanner(planner)
ctx = None


# observation and action spaces
# state is (x, theta, xdot, thetadot)
# use planner config for reasonable x-bounds and clamp angle to [-pi, pi]
_x_thr = getattr(cfg_planner, "x_threshold", None)
_x_low = -float(_x_thr)
_x_high = float(_x_thr)

#TODO: double check if these spaces are correct 
obs_low = np.array([_x_low, -np.pi, -np.inf, -np.inf], dtype=np.float32)
obs_high = np.array([_x_high, np.pi, np.inf, np.inf], dtype=np.float32)
obs_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

action_space = controller_wrapped.param_space

# SacZop config
cfg_saczop = SacTrainerConfig()


# Replay Buffer init
replay_buffer = ReplayBuffer(buffer_limit=cfg_saczop.buffer_size, device=device)


# critic init
extractor_cls = get_extractor_cls("identity")

critic = SacCritic(
    extractor_cls=extractor_cls,
    observation_space=obs_space,
    action_space=action_space,
    mlp_cfg=cfg_saczop.critic_mlp,
    num_critics=cfg_saczop.num_critics,
)

target_critic = SacCritic(
    extractor_cls=extractor_cls,
    observation_space=obs_space,
    action_space=action_space,
    mlp_cfg=cfg_saczop.critic_mlp,
    num_critics=cfg_saczop.num_critics,
)
target_critic.load_state_dict(critic.state_dict())


# actor init
actor = MpcSacActor(
    extractor_cls=extractor_cls,
    observation_space=obs_space,
    controller=controller_wrapped,
    distribution_name=cfg_saczop.distribution_name,
    mlp_cfg=cfg_saczop.actor_mlp,
    init_param_with_default=cfg_saczop.init_param_with_default,
)


# entropy temperature ALpha init
log_alpha = torch.nn.Parameter(
    torch.tensor(cfg_saczop.init_alpha, dtype=torch.float32).log()
)

alpha_optimizer = (
    torch.optim.Adam([log_alpha], lr=cfg_saczop.lr_alpha)
    if cfg_saczop.lr_alpha is not None
    else None
)

param_dim = int(np.prod(action_space.shape))
action_dim = 1
entropy_norm = param_dim / action_dim
target_entropy = -action_dim if cfg_saczop.target_entropy is None else cfg_saczop.target_entropy


# initializing optimizers
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg_saczop.lr_q)
actor_optimizer  = torch.optim.Adam(actor.parameters(),  lr=cfg_saczop.lr_pi)
