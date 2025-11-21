import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Normal

from leap_c.examples.cartpole.env import CartPoleBalanceEnv, CartPoleEnvConfig


class Policy(nn.Module):
    def __init__(self,
                 env,
                 sigma=0.1):
        super(Policy, self).__init__()
        self.env = env
        self.set_sigma(sigma)
        s_dim = env.observation_space.shape[0]
        a_dim = env.action_space.shape[0]
    
        self.features = nn.Linear(s_dim,
                                  a_dim,
                                  bias=False)

    def set_sigma(self, sigma):
        self.sigma = torch.full(self.env.action_space.shape, sigma)
        
    def forward(self, x, action=None):
        x = torch.as_tensor(x).float().flatten()
        x = self.features(x).reshape(self.sigma.shape)
        
        distr = Normal(x, self.sigma)  # The output of the features are used as the mean.

        if action is not None:
            return distr.log_prob(action)  # The log of the probabiliy density function.

        action = distr.sample()

        # We clip the action to ensure that they are in bound.
        return action.clip(self.env.action_space.low.item(),
                           self.env.action_space.high.item())

def plot_distribution(sigma):
    cfg = CartPoleEnvConfig()
    env = CartPoleBalanceEnv(render_mode="rgb_array", cfg=cfg)
    agent = Policy(env, sigma=sigma)

    state, _ = env.reset()
    actions = [agent(state).item() for _ in range(1000)]

    plt.hist(actions)
    plt.title("Action Distribution")
    plt.savefig(f"action_distribution_sigma_{sigma}.png")
    
if __name__ == "__main__":
    sigma=0.3
    plot_distribution(sigma)