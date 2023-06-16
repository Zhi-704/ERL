import numpy as np
import random
import gym
from gym import spaces
import pygame

Tetraminos = {
    'I': [(0, 0), (0, -1), (0, -2), (0, -3)],
    'L': [(0, 0), (1, 0), (0, -1), (0, -2)],
    'J': [(0, 0), (-1, 0), (0, -1), (0, -2)],
    'T': [(0, 0), (-1, 0), (1, 0), (0, -1)],
    'O': [(0, 0), (0, -1), (-1, 0), (-1, -1)],
    'S': [(0, 0), (-1, -1), (0, -1), (1, 0)],
    'Z': [(0, 0), (-1, 0), (0, -1), (1, -1)],
}
Tetraminos_name = ['I', 'L', 'J', 'T', 'O', 'S', 'Z']

class TetrisEnvDepreciated:
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode = None, width, height):
        self.width = width
        self.height = height
        self.board = np.zeros(shape=(width, height), dtype = int) # The size of tetris board
        self.window_size = 512  # The size of the PyGame window
        
        # We have 5 actions, corresponding to "left rotate", "right rotate", "move left", "move right", "drop"
        self.action_space = spaces.Discrete(5)

        # NEED TO IMPLEMENT
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def close(self):
    if self.window is not None:
        pygame.display.quit()
        pygame.quit()

    def _get_obs(self):
        """
        Translates environment's state into an observation.
        """
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        """
        Any other additional information needed or could implement such as manhattan distance. 
        Sometimes info will also contain data that is only available inside step such as reward steps
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        """
        Assume step method will not be called before reset has benncalled.
        Reset called whenever a done signal is issued
        Users may pass seed keyword to reset to initialise any random number generator used by the environment to a deterministic state
        if use self.np_random, you need to remember to called super().reset(seed=seed) to make sure gymnasiun.env correctly seeds RNG.
        When done, can randomly set state of environment
        For Tetris, randomly choose tetriminos but the starting placement is always the same.
        Method shuld return a tuple of initial obsercation and some auziliary information. Can use prior two method for that.
        """

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose random tetrinimos at random
        self.starting_piece = self.np_random.integers(0, 5, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        Accepts an action and computes state of environment after applying action and returns the 4-tuple (observation,reward,done,info).
        Once new state of environment is computed, can check whether it is a terminal state and we set done accordingly.
        """

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # An episode is done if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )