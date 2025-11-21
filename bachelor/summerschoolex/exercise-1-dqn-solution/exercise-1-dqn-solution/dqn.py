import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium
from collections import deque
import random
import numpy as np
from collections import namedtuple

import torch.optim as optim
from tqdm import trange

from visualization import plot_episode_stats, rendered_rollout
from replay_buffer import sample_batch

class Q(nn.Module):
    """ A Q-function network using 3 layers and relu activation functions."""
    def __init__(self, state_dim, action_dim, hidden_dim=50):
        super(Q, self).__init__()
        # TODO: Define the layers of the network here. 
        # Hint: Use nn.Linear
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.as_tensor(x).float()
        # TODO: Define the forward pass.
        # Hint: You can use "y = module(x)" to use the forward pass of a pytorch module like nn.Linear. 
        # Use F.relu for the activation functions.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])

class DQNAgent:
    def __init__(self, env, gamma, lr=0.001, maxlen=100000, batch_size=64):
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.env = env
        self.q = Q(state_dim, action_dim)
        self.q_target = Q(state_dim, action_dim)

        self.gamma = gamma
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.action_dim = action_dim
        self.batch_size = batch_size

        self.replay_buffer = deque(maxlen=maxlen)

    def get_action(self, x, epsilon):
        """ Epsilon-greedy action selection."""
        if np.random.uniform() < epsilon:
            return np.random.randint(self.action_dim)
        return self.q(x).argmax().item()

    def train(self, episodes, epsilon, update_freq=5, render_freq=40):
        stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes))

        training_steps = 0

        episode_range = trange(episodes)
        for e in episode_range:
            state, _ = self.env.reset()
            while True:
                action = self.get_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                stats.episode_rewards[e] += reward
                stats.episode_lengths[e] += 1

                # TODO: Add the samples to the replay buffer. Look at the code from above!
                self.replay_buffer.append((state, action, reward, next_state, terminated))  # add your code here :)

                # We sample a transition batch.
                bs, ba, br, bns, bterm = sample_batch(self.replay_buffer, self.batch_size)

                # TODO: Calculate the temporal difference target.
                # NOTE: The pytorch max function returns a tuple (values, indices)
                with torch.no_grad():
                    btarget = br + self.gamma * self.q_target(bns).max(1)[0] * (1-bterm)

                q_fwd = self.q(bs)
                current_prediction = q_fwd[torch.arange(self.batch_size).long(), ba.long()]
                loss = torch.square(current_prediction - btarget).mean()

                self.q_optimizer.zero_grad()  # Delete the outdated gradients.
                loss.backward()
                self.q_optimizer.step()

                if training_steps % update_freq == 0:
                    self.q_target.load_state_dict(self.q.state_dict())

                if truncated or terminated:
                    break

                state = next_state
                training_steps += 1

            if e % render_freq == 0:
                rendered_rollout(self, self.env, e)
            episode_range.set_description(f"Episode Reward {stats.episode_rewards[e]} ")
        rendered_rollout(self, self.env, episodes)
        return stats

def start_training():
    name = "CartPole-v1"
    env = gymnasium.make(name, render_mode="rgb_array", max_episode_steps=200)
    dqn = DQNAgent(env, gamma=0.99)

    stats = dqn.train(episodes=200, epsilon=0.1, update_freq=100)
    plot_episode_stats(stats, name)

if __name__ == "__main__":
    start_training()