import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image as IImage
from PIL import Image


def plot_episode_stats(stats, name, smoothing_window=10, noshow=False):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)

    # Plot the episode length over time
    ax = axes[0]
    ax.plot(stats.episode_lengths)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.set_title("Episode Length over Time")

    # Plot the episode reward over time
    ax = axes[1]
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ax.plot(rewards_smoothed)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward (Smoothed)")
    ax.set_title("Episode Reward over Time\n(Smoothed over window size {})".format(smoothing_window))
    plt.savefig(f"episode_stats_{name}.png")


def save_rgb_animation(rgb_arrays, filename, duration=500):
    """
    Save an animated GIF from a list of RGB arrays.

    Parameters:
        rgb_arrays (list of numpy arrays): List of RGB arrays (numpy arrays).
        filename (str): The name of the output GIF file.
        duration (int): Duration (in milliseconds) for each frame (default is 500).
    """
    # Create a list to hold each frame
    frames = []

    # Convert RGB arrays to PIL Image objects
    for rgb_array in rgb_arrays:
        img = Image.fromarray(rgb_array)
        frames.append(img)

    # Save the frames as an animated GIF
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=duration, loop=0)

def rendered_rollout(agent, env, episode, max_steps=200):
    current_sigma = agent.sigma
    # In the first exercise we rendered the explorative policy. Here we
    # set the exploration temporarily to almost zero to see the "real" policy
    agent.set_sigma(0.000001)
    steps = 0

    state, _ = env.reset()
    imgs = [env.render()]

    while steps < max_steps:
        action = agent(state)
        state, _, terminated, truncated, _ = env.step(action)
        imgs.append(env.render())
        
        steps += 1

        if terminated or truncated:
            break

    filename = f"{type(env).__name__}_episode_{episode}.gif"
    save_rgb_animation(imgs, filename, duration=40)
    IImage(filename=filename)
    agent.sigma = current_sigma