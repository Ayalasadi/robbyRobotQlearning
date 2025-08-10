#!/usr/bin/env python3
#hello, Utility functions for Q‑learning and epsilon scheduling
import numpy as np
import random


def epsilon_for_episode(episode: int, decay_interval: int = 50, initial_epsilon: float = 0.1) -> float:
    """Compute epsilon for ε‑greedy action selection.

    Epsilon is decreased by a fixed step every `decay_interval` episodes until it reaches zero.

    Args:
        episode: Current episode number (0‑based).
        decay_interval: Number of episodes between epsilon decrements.
        initial_epsilon: Starting epsilon value.

    Returns:
        The epsilon value for this episode.
    """
    #hello, compute how many full decay intervals have passed
    decay_steps = episode // decay_interval
    #hello, compute epsilon decrement step so that it hits zero after 10 intervals if initial_epsilon is 0.1
    decrement = initial_epsilon / 10.0 if initial_epsilon > 0 else 0.0
    eps = max(0.0, initial_epsilon - decrement * decay_steps)
    return eps


def select_action(Q: np.ndarray, state: int, epsilon: float, rng: random.Random | None = None) -> int:
    """Select an action index using ε‑greedy strategy.

    Args:
        Q: Q‑table of shape (num_states, num_actions).
        state: Current state index.
        epsilon: Probability of choosing a random action.
        rng: Optional random number generator for reproducibility.

    Returns:
        Action index.
    """
    #hello, use provided RNG or numpy for random choice
    if rng is None:
        rng = random.Random()
    if rng.random() < epsilon:
        return rng.randrange(Q.shape[1])
    #hello, choose the greedy action; break ties randomly
    values = Q[state]
    max_value = np.max(values)
    candidate_actions = [i for i, v in enumerate(values) if v == max_value]
    return rng.choice(candidate_actions)