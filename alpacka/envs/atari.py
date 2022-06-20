"""Atari environments."""

import copy

import gym
import numpy as np
from gym.wrappers import atari_preprocessing

from alpacka.envs import base


try:
    from gym.envs.atari import atari_env  # pylint: disable=ungrouped-imports
except gym.error.DependencyNotInstalled:
    atari_env = None


class Atari(base.RestorableEnv):
    """Atari env with DeepMind wrappers."""

    installed = atari_env is not None

    allowed_stochasticities = [
        base.Stochasticity.none,
        base.Stochasticity.episodic,
        base.Stochasticity.universal,
    ]

    class Renderer(base.EnvRenderer):
        """Renderer for Atari."""

        def render_state(self, state_info):
            obs = np.array(state_info) * 255
            return np.broadcast_to(obs, obs.shape[:2] + (3,)).astype(np.uint8)

        def render_action(self, action):
            return atari_env.ACTION_MEANING[action].lower()

    def __init__(
        self,
        game='pong',
        stochasticity=base.Stochasticity.episodic,
        sticky_actions=None,
        env_kwargs=None,
        wrapper_kwargs=None,
    ):
        """Initializes an Atari env.

        Applies the DeepMind wrapper to the env, which adds, among others, the
        following transformations:

            - Sticky actions: repeating the last action with some probability.
                Used to add stochasticity to the environment. Disabled by
                default, unless the stochasticity mode is "universal".
            - Initial no-ops: a random number of initial no-op actions to take
                on the env. Used to average over multiple different trajectories
                during evaluation, but also doesn't hurt during training.
                Enabled by default, unless the stochasticity mode is "none".
            - Frame skip & max pooling: to decrease the time horizon, we show
                the agent every fourth frame. The observation is a maximum of
                pixel values over last two frames.

        Other transformations can be enabled using wrapper_kwargs. The defaults
        follow the suggestions of Machado et al. - Revisiting the Arcade
        Learning Environment: Evaluation Protocols and Open Problems for General
        Agents (2017).

        Args:
            game (str): Identifier of the game, camel_case.
            stochasticity (envs.Stochasticity): Stochasticity mode.
            sticky_actions (bool): Whether to use sticky actions. The default
                depends on the stochasticity mode.
            env_kwargs (dict): Kwargs forwarded to AtariEnv.
            wrapper_kwargs (dict): Kwargs forwarded to the AtariPreprocessing
                wrapper.

        Raises:
            ImportError: When gym[atari] has not been installed.
        """
        if atari_env is None:
            raise ImportError(
                'Could not import gym.envs.atari! HINT: Install gym[atari].'
            )

        if sticky_actions is None:
            # In the universal stochasticity mode, use sticky actions by default
            # - otherwise we likely won't see any stochasticity at all.
            sticky_actions = stochasticity is base.Stochasticity.universal

        default_repeat_prob = 0.0
        if stochasticity is base.Stochasticity.none:
            # Disable the stochastic modifications.
            default_noop_max = 0
            assert not sticky_actions
        else:
            # Defaults suggested by Machado et al.
            default_noop_max = 30
            if sticky_actions:
                default_repeat_prob = 0.25

        env_kwargs = {
            'obs_type': 'image',
            'frameskip': 1,
            'repeat_action_probability': default_repeat_prob,
            **(env_kwargs or {})
        }
        env = atari_env.AtariEnv(game, **env_kwargs)

        # Simulate that the env has been created via gym.make, so by a string
        # ID. The 'NoFrameskip' mode is required by the wrapper.
        class Spec:
            id = 'NoFrameskip'
        env.spec = Spec

        wrapper_kwargs = {
            'noop_max': default_noop_max,
            'scale_obs': True,
            **(wrapper_kwargs or {})
        }
        self._env = atari_preprocessing.AtariPreprocessing(
            env, **wrapper_kwargs
        )
        self.stochasticity = stochasticity

        self.observation_space = copy.copy(self._env.observation_space)
        if len(self.observation_space.shape) == 2:
            # Add the depth dimension.
            self.observation_space = gym.spaces.Box(
                low=self.observation_space.low[..., None],
                high=self.observation_space.high[..., None],
            )
        self.action_space = self._env.action_space

    def reset(self):
        self._env.reset()
        return self._observation

    def step(self, action):
        (_, reward, done, info) = self._env.step(action)
        return (self._observation, reward, done, info)

    def close(self):
        self._env.close()

    def clone_state(self):
        if self.stochasticity is base.Stochasticity.universal:
            env_state = self._env.clone_state()
        else:
            env_state = self._env.clone_full_state()

        # Include the max pooling buffer in the state.
        state = (env_state, copy.deepcopy(self._env.obs_buffer))

        if self.stochasticity is not base.Stochasticity.universal:
            state += (self._env.unwrapped.np_random,)

        return state

    def restore_state(self, state):
        (env_state, obs_buffer) = state[:2]

        if self.stochasticity is base.Stochasticity.universal:
            self._env.restore_state(env_state)
        else:
            self._env.restore_full_state(env_state)
            self._env.unwrapped.np_random = state[2]

        self._env.obs_buffer = copy.deepcopy(obs_buffer)
        return self._observation

    @property
    def state_info(self):
        # Just return the observation. Picture is worth a thousand words.
        return self._observation

    @property
    def _observation(self):
        obs = self._env._get_obs()  # pylint: disable=protected-access
        if len(obs.shape) == 2:
            # Add the depth dimension.
            obs = obs[..., None]
        return obs.copy()
