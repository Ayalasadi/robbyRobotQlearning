#!/usr/bin/env python3
#this module defines the Robby grid environment for a simple Q-learning agent
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
        #store grid size and can probability
        self.grid_size = grid_size
        self.can_prob = can_prob
        #use a dedicated random.Random for reproducibility
        self.random = random.Random(seed)
        #define possible sensor values
        self.EMPTY = 0
        self.CAN = 1
        self.WALL = 2
        #define action indices
        self.ACTIONS = ['N', 'S', 'E', 'W', 'PICK']
        #initialize grid and position
        self.grid: np.ndarray | None = None
        self.position: tuple[int, int] | None = None

    #backward-compat alias so code that uses env.agent_pos still works
    @property
    def agent_pos(self) -> tuple[int, int]:
        assert self.position is not None
        return self.position

    @agent_pos.setter
    def agent_pos(self, value: tuple[int, int]) -> None:
        self.position = value

    #backward-compat alias for older code calling _observe_state()
    def _observe_state(self) -> int:
        return self._encode_state()

    def reset(self) -> int:
        """Reset the environment and return the initial encoded state."""
        #generate a new grid with cans placed independently
        #create a random boolean grid where True indicates a can
        self.grid = (np.random.rand(self.grid_size, self.grid_size) < self.can_prob).astype(int)
        #choose a random starting position
        x = self.random.randrange(self.grid_size)
        y = self.random.randrange(self.grid_size)
        self.position = (x, y)
        #return initial state encoding
        return self._encode_state()

    def _get_sensor_values(self) -> tuple[int, int, int, int, int]:
        """Return a tuple of sensor values (current, north, south, east, west)."""
        assert self.position is not None and self.grid is not None
        x, y = self.position

        #helper to check cell content or wall
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
        #compute a base-3 number from sensor values
        s0, s1, s2, s3, s4 = self._get_sensor_values()
        return s0 + 3 * s1 + 9 * s2 + 27 * s3 + 81 * s4

    #action indices: 0=N, 1=S, 2=E, 3=W, 4=Pick
    def step(self, action: int) -> tuple[int, float]:
        #make sure env is initialized
        assert self.position is not None and self.grid is not None

        #default reward and unpack current position
        reward = 0.0
        r, c = self.position

        if action == 0:  #move-north
            if r == 0:
                reward = -5.0
            else:
                r -= 1
        elif action == 1:  #move-south
            if r == self.grid_size - 1:
                reward = -5.0
            else:
                r += 1
        elif action == 2:  #move-east
            if c == self.grid_size - 1:
                reward = -5.0
            else:
                c += 1
        elif action == 3:  #move-west
            if c == 0:
                reward = -5.0
            else:
                c -= 1
        elif action == 4:  #pick-up-can
            if self.grid[r, c] == 1:
                reward = 10.0
                self.grid[r, c] = 0
            else:
                reward = -1.0
        else:
            raise ValueError(f"Unknown action index: {action}")

        #apply movement only if we didn’t hit a wall
        if action in (0, 1, 2, 3) and reward != -5.0:
            self.position = (r, c)

        #return next state and reward
        return self._encode_state(), float(reward)
