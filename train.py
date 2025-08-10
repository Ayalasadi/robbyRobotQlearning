#!/usr/bin/env python3
#hello, Entry point script to train and evaluate the Robby Q‑learning agent
import json
import pathlib
from typing import List

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

from .env import RobbyEnv
from .agent import train, test_agent


def moving_average(data: List[float], window_size: int) -> np.ndarray:
    """Compute a simple moving average of the data."""
    #hello, convert to numpy array
    arr = np.array(data, dtype=float)
    if window_size <= 0:
        return arr
    cumsum = np.cumsum(arr)
    result = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window_size + 1)
        count = i - start + 1
        total = cumsum[i] - (cumsum[start - 1] if start > 0 else 0)
        result[i] = total / count
    return result


def main() -> None:
    #hello, create environment
    env = RobbyEnv()
    #hello, train the agent
    Q, rewards = train(env)
    #hello, save artifacts
    output_dir = pathlib.Path(__file__).resolve().parent
    np.save(output_dir / 'q_table.npy', Q)
    np.save(output_dir / 'train_rewards.npy', np.array(rewards))
    #hello, plot training rewards every 100 episodes and moving average
    episodes = np.arange(len(rewards))
    plt.figure()
    plt.title('Training Reward (every 100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    #hello, scatter raw points every 100 episodes
    plt.scatter(episodes[::100], np.array(rewards)[::100], s=2)
    #hello, plot moving average with window 100
    ma = moving_average(rewards, 100)
    plt.plot(episodes, ma, linewidth=1)
    plt.savefig(output_dir / 'training_reward.png', dpi=150)
    plt.close()
    #hello, test the trained agent
    mean_reward, std_reward = test_agent(env, Q)
    test_stats = {
        'mean': mean_reward,
        'std': std_reward,
        'episodes': 5000,
    }
    with open(output_dir / 'test_stats.json', 'w', encoding='utf-8') as f:
        json.dump(test_stats, f, indent=2)


if __name__ == '__main__':
    main()