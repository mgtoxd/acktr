import sys

import gym
from gym import spaces
import pygame
import numpy as np

from server.Core import CoreChoose


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=5, Core: CoreChoose = None):
        if Core is not None:
            self.Core = Core
            self.observation_space = spaces.Box(0, sys.maxsize, shape=(1, len(self.Core.sizeMap), 3), dtype=int)
            self.action_space = spaces.Discrete(len(self.Core.sizeMap))
            self.ob = np.zeros((len(self.Core.sizeMap), 3))[np.newaxis, :]
            self.reward = 0
            self.delay = 0
            self.hash2Index = {}
            idx = 0
            for key in self.Core.sizeMap.keys():
                self.hash2Index[key] = idx
                self.hash2Index[idx] = key
                idx += 1
        else:

            self.size = size  # The size of the square grid
            self.window_size = 512  # The size of the PyGame window
            self.reward = 0
            # Observations are dictionaries with the agent's and the target's location.
            # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
            self.observation_space = spaces.Box(0, 1000, shape=(1, size, 3), dtype=int)

            # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
            self.action_space = spaces.Discrete(size)

            self.ob = np.random.randint(0, 1000, size=[size, 3])[np.newaxis, :]

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

            """
            If human-rendering is used, `self.window` will be a reference
            to the window that we draw to. `self.clock` will be a clock that is used
            to ensure that the environment is rendered at the correct framerate in
            human-mode. They will remain `None` until human-mode is used for the
            first time.
            """
            self.window = None
            self.clock = None

    def _get_obs(self):
        return self.ob

    def _get_info(self):
        return {'episode': {
            'r': self.reward
        }}

    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        # super().reset()

        # Choose the agent's location uniformly at random
        # self._agent_location = self.np_random.integers(0, self.size, size=2)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(0, self.size, size=2)

        observation = self._get_obs()
        info = self.hash2Index
        return (observation, info) if return_info else observation

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        # direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        # self._agent_location = np.clip(
        #     self._agent_location + direction, 0, self.size - 1
        # )
        # An episode is done iff the agent has reached the target
        # done = np.array_equal(self._agent_location, self._target_location)
        # reward = 1 if done else 0  # Binary sparse rewards
        # observation = self._get_obs()
        # info = self._get_info()
        if self.Core is not None:
            self.ob = np.zeros((len(self.Core.sizeMap), 3))[np.newaxis, :]
            currdelay = 0
            for key, v in self.Core.delayMap.items():
                currdelay += v['delay']
                self.ob[0][self.hash2Index[key]][0] = v['delay']
                self.ob[0][self.hash2Index[key]][1] = v['count']
                self.ob[0][self.hash2Index[key]][2] = self.Core.sizeMap[key]
            if self.delay == 0:
                self.reward = 0
            else:
                self.reward = (self.delay - currdelay) / self.delay
            observation = self._get_obs()
            info = self._get_info()
            done = False

        else:
            self.reward = self.ob[0][action][0]
            self.ob[0][action][0] += 10
            done = False
            observation = self._get_obs()
            info = self._get_info()

        return observation, self.reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass
