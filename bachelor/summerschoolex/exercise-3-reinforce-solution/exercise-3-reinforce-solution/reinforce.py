from collections import namedtuple

import torch
from policy import Policy
from tqdm import trange
from visualization import plot_episode_stats, rendered_rollout

from leap_c.examples.cartpole.env import CartPoleBalanceEnv, CartPoleEnvConfig

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def rollout(agent, env, render=False, sigma=0.):
    """ Collects states, rewards and actions of one episode."""

    rewards = []
    actions = []
    states = []

    state, _ = env.reset()

    while True:
        state = torch.tensor(state).float()
        states.append(state)

        with torch.no_grad():
            action = agent(state)

        state, reward, truncated, terminated, _ = env.step(action)

        rewards.append(reward)
        actions.append(action)

        if truncated or terminated:
            break

    return states, torch.tensor(rewards).float(), actions


def reinforce(agent, env, num_episodes=1000, gamma=0.99, lr=0.0001):
    """ A short implementation of reinforce."""
    render_freq = 1000
    optimizer = torch.optim.SGD(agent.parameters(), lr=lr)

    episode_rewards = []
    episode_lengths = []

    episode_range = trange(num_episodes)
    for episode in episode_range:
        states, rewards, actions = rollout(agent, env)
        num_steps = len(rewards)

        for t, (state, action) in enumerate(zip(states, actions)):
            state = torch.as_tensor(state).to(torch.float)
            
            # TODO 1: Calculate the disc return of the episode
            # Tipp: torch.inner and torch.arange are nice functions for this :)
            disc_return = torch.inner(rewards[t:num_steps], gamma**torch.arange(num_steps - t))
            
            # TODO 2: Calculate the policy_objective.
            policy_objective = gamma**t * disc_return * agent(state, action)

            optimizer.zero_grad()
            (-policy_objective.mean()).backward()  # We do gradient ASCENT not descent!
            optimizer.step()
        
        episode_rewards.append(sum(rewards))
        episode_lengths.append(num_steps)
        episode_range.set_description(f"Episode Reward {episode_rewards[-1].item():4} ")
        
        if episode % render_freq == 0:
            rendered_rollout(agent, env, episode)
    rendered_rollout(agent, env, num_episodes) # Render final policy
    
    return EpisodeStats(episode_lengths, episode_rewards)

def start_training():
    name = "ContCartPoleBalance"
    cfg = CartPoleEnvConfig()
    env = CartPoleBalanceEnv(render_mode="rgb_array", cfg=cfg)
    agent = Policy(env, sigma=0.4)
    stats = reinforce(agent, env, lr=0.001, num_episodes=5000)
    plot_episode_stats(stats, name)
    
if __name__ == "__main__":
    start_training()