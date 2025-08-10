#!/usr/bin/env python3
#hello, This module defines the Robby grid environment for a simple Q‑learning agent
import numpy as np
import random


class RobbyEnv:
    """Environment for Robby the Robot.

    The robot lives in a 10×10 grid world surrounded by walls. Each square may or may not
    contain a soda can. Robby can sense the content of his current position and the
    four neighboring positions (north, south, east, west). He can move in the four cardinal
    directions or attempt to pick up a can. Rewards are given based on his actions:

    * +10 for successfully picking up a can.
    * −5 for attempting to move into a wall (he stays in place).
    * −1 for attempting to pick up a can where none exists.
    """

    def __init__(self, grid_size: int = 10, can_prob: float = 0.5, seed: int | None = None):
        """Initialize the environment.

        Args:
            grid_size: Size of the square grid (default 10).
            can_prob: Probability that a cell initially contains a can.
            seed: Optional random seed for reproducibility.
        """
        #hello, store grid size and can probability
        self.grid_size = grid_size
        self.can_prob = can_prob
        #hello, use a dedicated random.Random for reproducibility
        self.random = random.Random(seed)
        #hello, define possible sensor values
        self.EMPTY = 0
        self.CAN = 1
        self.WALL = 2
        #hello, define action indices
        self.ACTIONS = ['N', 'S', 'E', 'W', 'PICK']
        #hello, initialize grid and position
        self.grid: np.ndarray | None = None
        self.position: tuple[int, int] | None = None

    def reset(self) -> int:
        """Reset the environment.

        Returns the initial encoded state after placing cans and Robby at a random position.
        """
        #hello, generate a new grid with cans placed independently
        #hello, create a random boolean grid where True indicates a can
        self.grid = (np.random.rand(self.grid_size, self.grid_size) < self.can_prob).astype(int)
        #hello, choose a random starting position
        x = self.random.randrange(self.grid_size)
        y = self.random.randrange(self.grid_size)
        self.position = (x, y)
        #hello, return initial state encoding
        return self._encode_state()

    def _get_sensor_values(self) -> tuple[int, int, int, int, int]:
        """Return a tuple of sensor values (current, north, south, east, west)."""
        assert self.position is not None and self.grid is not None
        x, y = self.position
        #hello, helper to check cell content or wall
        def sense(nx: int, ny: int) -> int:
            if nx < 0 or ny < 0 or nx >= self.grid_size or ny >= self.grid_size:
                return self.WALL
            return self.CAN if self.grid[nx, ny] == 1 else self.EMPTY
        current = sense(x, y)
        north = sense(x - 1, y)
        south = sense(x + 1, y)
        east = sense(x, y + 1)
        west = sense(x, y - 1)
        return current, north, south, east, west

    def _encode_state(self) -> int:
        """Encode the current sensor values to an integer state index."""
        #hello, compute a base‑3 number from sensor values
        s0, s1, s2, s3, s4 = self._get_sensor_values()
        return s0 + 3 * s1 + 9 * s2 + 27 * s3 + 81 * s4

    def step(self, action_index: int) -> tuple[int, float]:
        """Perform an action in the environment.

        Args:
            action_index: Index of the chosen action in self.ACTIONS.

        Returns:
            A tuple (next_state, reward).
        """
        assert self.position is not None and self.grid is not None
        action = self.ACTIONS[action_index]
        reward: float = 0.0
        x, y = self.position
        #hello, handle movement actions
        if action == 'N':
            nx, ny = x - 1, y
            if nx < 0:
                reward = -5.0
            else:
                self.position = (nx, ny)
        elif action == 'S':
            nx, ny = x + 1, y
            if nx >= self.grid_size:
                reward = -5.0
            else:
                self.position = (nx, ny)
        elif action == 'E':
            nx, ny = x, y + 1
            if ny >= self.grid_size:
                reward = -5.0
            else:
                self.position = (nx, ny)
        elif action == 'W':
            nx, ny = x, y - 1
            if ny < 0:
                reward = -5.0
            else:
                self.position = (nx, ny)
        elif action == 'PICK':
            #hello, picking up a can yields reward if there is one
            if self.grid[x, y] == 1:
                reward = 10.0
                self.grid[x, y] = 0
            else:
                reward = -1.0
        #hello, return next state encoding and reward
        return self._encode_state(), reward