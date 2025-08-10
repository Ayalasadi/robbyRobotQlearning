# Robby Q‑learning Project

This project contains a simple implementation of **Robby the Robot** using Q‑learning. The robot lives in a 10×10 grid world and learns to pick up soda cans while avoiding walls.

## Files

- `env.py`: Defines the `RobbyEnv` class representing the grid world environment.
- `utils.py`: Provides helper functions for the ε‑greedy policy and epsilon scheduling.
- `agent.py`: Implements the training (`train`) and testing (`test_agent`) routines for the Q‑learning agent.
- `train.py`: Script to run training and evaluation, save the results, and plot the training reward.
- `q_table.npy`: Saved NumPy array of the learned Q‑table (generated after training).
- `train_rewards.npy`: Saved training rewards per episode (generated after training).
- `training_reward.png`: Plot of the cumulative reward per episode (generated after training).
- `test_stats.json`: JSON file with mean and standard deviation of rewards from the test phase.

## Requirements

The code has been tested with Python 3.10 and requires the following packages:

- `numpy`
- `matplotlib`

You can install dependencies with:

```bash
pip install numpy matplotlib
```

## Usage

1. **Train and evaluate the agent**:

```bash
python -m robby_q_learning.train
```

This will train the agent for 5,000 episodes with 200 steps per episode, save the Q‑table and reward history, generate a plot of the training rewards, and write test statistics to a JSON file.

2. **Modify parameters**:

The default parameters in `train.py` can be adjusted by editing the `train` and `test_agent` function calls in `main()`. For example, you can change the number of episodes, learning rate, discount factor, or epsilon schedule.

## Repository setup

This project is intended to be version‑controlled. After creating the folder and files, you can initialize a Git repository and push the code to GitHub.

```bash
cd robby_q_learning
git init
git add .
git commit -m "Initial commit"
git remote add origin <your‑repository‑url>
git push -u origin main
```

Alternatively, you can use the GitHub API to create a new repository and upload the files directly.