"""Sokoban environments."""

import copy
import random

import numpy as np

from alpacka.envs import base


try:
    from gym_sokoban.envs import sokoban_env_fast
except ImportError:
    sokoban_env_fast = None


class Sokoban(base.RestorableEnv):
    """Sokoban with state clone/restore and returning a "solved" flag.

    Returns observations in one-hot encoding.
    """

    installed = sokoban_env_fast is not None
    stochasticity = base.Stochasticity.episodic

    class Renderer(base.EnvRenderer):
        """Renderer for Sokoban."""

        def __init__(self, env):
            """Initializes render surfaces."""
            super().__init__(env)

            self._render_surfaces = env._env.unwrapped._surfaces['16x16pixels']

        def render_state(self, state_info):
            onehot_state = np.array(state_info)

            size_x = onehot_state.shape[0] * self._render_surfaces.shape[1]
            size_y = onehot_state.shape[1] * self._render_surfaces.shape[2]

            # onehot_state is (board_size_x, board_size_y, 7),
            # where 7 stands for 7 types of fields (box, empty, etc).
            # self._render_surfaces is (7, img_x, img_y, 3) being
            # depictions for all of the surface types.
            img = np.tensordot(onehot_state, self._render_surfaces, (-1, 0))
            # img is now (board_size_x, board_size_y, img_x, img_y, 3).
            img = np.transpose(img, (0, 2, 1, 3, 4))
            # img is now (board_size_x, img_x, board_size_y, img_y, 3).
            img = np.reshape(img, (size_x, size_y, 3))

            return img.astype(np.uint8)

        def render_action(self, action):
            return ['up', 'down', 'left', 'right'][action]

    def __init__(self, *args, **kwargs):
        super().__init__()

        if sokoban_env_fast is None:
            raise ImportError(
                'Could not import Sokoban. Install alpacka[sokoban].'
            )
        self._env = sokoban_env_fast.SokobanEnvFast(*args, **kwargs)

        self.observation_space = copy.deepcopy(self._env.observation_space)
        # Return observations as float32, so we don't have to cast them in the
        # network training pipeline.
        self.observation_space.dtype = np.float32

        self.action_space = self._env.action_space

    def reset(self):
        return self._env.reset().astype(np.float32)

    def step(self, action):
        (observation, reward, done, info) = self._env.step(action)
        return (observation.astype(np.float32), reward, done, info)

    def clone_state(self):
        return self._env.clone_full_state()

    def restore_state(self, state):
        self._env.restore_full_state(state)
        return self._env.render(mode=self._env.mode)


class ActionNoiseSokoban(Sokoban):
    """Sokoban with randomized actions."""

    stochasticity = base.Stochasticity.universal

    def __init__(self, action_noise, *args, **kwargs):
        """Initializes ActionNoiseSokoban.

        Args:
            action_noise: float, how often action passed to step() should be
                replaced by one sampled uniformly from action space.
            args: passed to Sokoban.__init__()
            kwargs: passed to Sokoban.__init__()
        """
        super().__init__(*args, **kwargs)
        self._action_noise = action_noise

    def step(self, action):
        if random.random() < self._action_noise:
            action = self.action_space.sample()
        return super().step(action)
