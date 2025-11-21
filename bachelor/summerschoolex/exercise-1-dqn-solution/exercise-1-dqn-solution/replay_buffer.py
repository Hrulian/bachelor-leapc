import torch
import random
import gymnasium
from collections import deque, namedtuple

def sample_batch(replay_buffer, batch_size):
    return [torch.tensor(e, dtype=torch.float) for e in zip(*random.choices(replay_buffer, k=batch_size))]

def example_replay_buffer():
    env =  gymnasium.make("MountainCar-v0")
    rb = deque(maxlen=30)  # First in first out queue.

    # For demonstration purposes we fill the replay buffer by choosing 100 random actions.
    state = env.reset()

    for _ in range(100):
        action = env.action_space.sample()  # We choose a random action
        next_state, reward, terminated, truncated, _ = env.step(action)

        rb.append([state, action, reward, next_state, terminated])
        state = next_state
    batch = sample_batch(rb, 5)
    print("Random Batch with size 5: ")
    print("State:", batch[0])
    print("Action:", batch[1])
    print("Reward:", batch[2])
    print("Next State:", batch[3])
    print("Terminal:", batch[4])

if __name__ == "__main__":
    example_replay_buffer()