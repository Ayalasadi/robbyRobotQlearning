#!/usr/bin/env python3
#hello, Implementation of the Q‑learning agent for the Robby environment
import numpy as np
import random
from typing import Tuple, List

from .env import RobbyEnv
from .utils import epsilon_for_episode, select_action


def train(
    env: RobbyEnv,
    num_episodes: int = 5000,
    steps_per_episode: int = 200,
    alpha: float = 0.2,
    gamma: float = 0.9,
    epsilon_init: float = 0.1,
    decay_interval: int = 50,
    seed: int | None = None,
) -> Tuple[np.ndarray, List[float]]:
    """Train a Q‑learning agent on the given environment.

    Args:
        env: Instance of RobbyEnv.
        num_episodes: Total number of training episodes.
        steps_per_episode: Number of steps per episode.
        alpha: Learning rate η.
        gamma: Discount factor γ.
        epsilon_init: Initial epsilon for ε‑greedy policy.
        decay_interval: Episodes between epsilon decrements.
        seed: Optional random seed for reproducibility.

    Returns:
        A tuple (Q_table, rewards_per_episode).
    """
    #hello, initialize Q‑table with zeros (243 states × 5 actions)
    Q = np.zeros((243, 5), dtype=float)
    #hello, list to record cumulative reward per episode
    rewards_history: List[float] = []
    #hello, random generator for reproducibility
    rng = random.Random(seed)
    #hello, iterate over episodes
    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        epsilon = epsilon_for_episode(ep, decay_interval, epsilon_init)
        for _ in range(steps_per_episode):
            action = select_action(Q, state, epsilon, rng)
            next_state, reward = env.step(action)
            total_reward += reward
            #hello, Q‑learning update rule
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])
            state = next_state
        rewards_history.append(total_reward)
    return Q, rewards_history


def test_agent(
    env: RobbyEnv,
    Q: np.ndarray,
    num_episodes: int = 5000,
    steps_per_episode: int = 200,
    epsilon: float = 0.1,
    seed: int | None = None,
) -> Tuple[float, float]:
    """Test a trained agent by running episodes with a fixed epsilon.

    Args:
        env: Robby environment instance.
        Q: Trained Q‑table.
        num_episodes: Number of test episodes.
        steps_per_episode: Number of steps per episode.
        epsilon: Epsilon value to use during testing.
        seed: Optional seed for reproducibility.

    Returns:
        A tuple (mean_reward, std_reward) of per‑episode total rewards.
    """
    #hello, list for tracking rewards
    rewards: List[float] = []
    rng = random.Random(seed)
    for _ in range(num_episodes):
        state = env.reset()
        total = 0.0
        for _ in range(steps_per_episode):
            action = select_action(Q, state, epsilon, rng)
            next_state, reward = env.step(action)
            total += reward
            state = next_state
        rewards.append(total)
    #hello, compute statistics
    return float(np.mean(rewards)), float(np.std(rewards))