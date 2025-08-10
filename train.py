#!/usr/bin/env python3
#entry point script to train and evaluate the Robby Q-learning agent
import json
import argparse
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from .env import RobbyEnv
from .agent import train, test_agent


#simple moving average for plotting
def moving_average(x, w):
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) == 0:
        return x
    c = np.cumsum(x)
    c[w:] = c[w:] - c[:-w]
    return c[w - 1:] / w


def run(args):
    #make outputs land in the current working directory
    out_dir = pathlib.Path(".").resolve()

    #hyperparams per assignment
    alpha = 0.2
    gamma = 0.9

    #env + train
    env = RobbyEnv(seed=args.seed)
    Q, rewards = train(
        env,
        num_episodes=args.episodes,
        steps_per_episode=args.steps,
        alpha=alpha,
        gamma=gamma,
        epsilon_init=0.1,
        decay_interval=50,
        seed=args.seed,
    )

    #save artifacts
    np.save(out_dir / "q_table.npy", Q)
    np.save(out_dir / "train_rewards.npy", np.asarray(rewards, dtype=float))

    #plot training reward: raw points every 100 eps + moving average
    fig = plt.figure()
    xs = np.arange(len(rewards))
    pts_x = xs[xs % 100 == 0]
    pts_y = np.asarray(rewards)[xs % 100 == 0]
    plt.scatter(pts_x, pts_y, s=8)
    ma = moving_average(rewards, 100)
    if len(ma) > 0:
        plt.plot(np.arange(99, 99 + len(ma)), ma)
    plt.xlabel("episode")
    plt.ylabel("sum of rewards")
    plt.title(f"Training Reward (seed={args.seed})")
    fig.tight_layout()
    fig.savefig(out_dir / "training_reward.png")
    plt.close(fig)

    #test with epsilon=0.1
    mean, std = test_agent(
        env,
        Q,
        num_episodes=args.episodes,   #match the actual parameter name
        steps_per_episode=args.steps, #match the actual parameter name
        epsilon=0.1,
        seed=(None if args.seed is None else args.seed + 1),
)

    #save + print test stats
    stats = {"episodes": int(args.episodes), "mean": float(mean), "std": float(std)}
    (out_dir / "test_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"test episodes: {stats['episodes']}")
    print(f"test mean: {stats['mean']:.2f}")
    print(f"test std: {stats['std']:.2f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)