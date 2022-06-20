"""Google Football environment."""

import collections
import copy

import gym
import numpy as np

from alpacka.envs import base
from alpacka.third_party import gfootball as gfootball_utils


try:
    import gfootball.env as football_env
except ImportError:
    football_env = None


class GoogleFootball(base.RestorableEnv):
    """Google Research Football conforming to the RestorableEnv interface."""

    installed = football_env is not None
    state_size = 480000

    stochasticity = base.Stochasticity.episodic

    class Renderer(base.EnvRenderer):
        """Renderer for GFootball.

        Uses the state visualization from GFootball's dumped videos.
        """

        def render_state(self, state_info):
            return gfootball_utils.get_frame(state_info)

        def render_action(self, action):
            action_set = football_env.football_action_set.full_action_set
            return str(action_set[action])


    def __init__(self,
                 env_name='academy_empty_goal_close',
                 representation='simple115',
                 rewards='scoring,checkpoints',
                 stacked=False,
                 dump_path=None,
                 solved_at=1,
                 **kwargs):
        if football_env is None:
            raise ImportError('Could not import gfootball! '
                              'HINT: Perform the setup instructions here: '
                              'https://github.com/google-research/football')

        self._solved_at = solved_at
        self._env = football_env.create_environment(
            env_name=env_name,
            representation=representation,
            rewards=rewards,
            stacked=stacked,
            write_full_episode_dumps=dump_path is not None,
            write_goal_dumps=False,
            logdir=dump_path or '',
            **kwargs
        )

        self.action_space = self._env.action_space
        self.observation_space = copy.copy(self._env.observation_space)
        self.observation_space.dtype = np.float32
        assert len(self.observation_space.shape) in (1, 3), (
            'Unsupported observation shape: {}.'.format(
                self.observation_space.shape
            )
        )
        self._has_pixel_input = len(self.observation_space.shape) == 3

    def _scale_obs(self, obs):
        if self._has_pixel_input:
            # Pixel observations are given as uint8. Cast to float and scale
            # to the [0, 1) interval.
            return (obs.astype(np.float32) - 128) / 128
        else:
            # Float observations don't need scaling.
            return obs

    def reset(self):
        # pylint: disable=protected-access
        obs = self._env.reset()
        env = self._env.unwrapped
        env._env._trace._trace = collections.deque([], 4)

        return self._scale_obs(obs)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        if done:
            info['solved'] = info['score_reward'] >= self._solved_at
        return self._scale_obs(obs), reward, done, info

    def render(self, mode='human'):
        return self._env.render(mode)

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        raise NotImplementedError

    def clone_state(self):
        # pylint: disable=protected-access
        raw_state = self._env.get_state()
        size_encoded = len(raw_state).to_bytes(3, byteorder='big')
        # Byte suffix to enforce self.state_size of state.
        suffix = bytes(self.state_size - len(size_encoded) - len(raw_state))
        resized_state = size_encoded + raw_state + suffix
        return np.frombuffer(resized_state, dtype=np.uint8)

    def restore_state(self, state):
        assert state.size == self.state_size, (
            f'State size does not match: {state.size} != {self.state_size}')

        # First 3 bytes encodes size of state.
        size_decoded = int.from_bytes(list(state[:3]), byteorder='big')
        raw_state = state[3:(size_decoded + 3)]
        assert (state[(size_decoded + 3):] == 0).all()

        self._env.set_state(bytes(raw_state))

        # Temporary fix for a bug in Football related to restoring state after
        # done. Proper fix is on the way:
        # https://github.com/google-research/football/pull/135
        # pylint: disable=protected-access,import-outside-toplevel,no-name-in-module,import-error
        from gfootball.env import observation_processor
        env = self._env.unwrapped._env
        if env._trace is None:
            env._trace = observation_processor.ObservationProcessor(env._config)

        return self._scale_obs(self._observation)

    @property
    def _observation(self):
        # pylint: disable=protected-access
        observation = self._env.unwrapped._env.observation()
        observation = self._env.unwrapped._convert_observations(
            observation, self._env.unwrapped._agent,
            self._env.unwrapped._agent_left_position,
            self._env.unwrapped._agent_right_position
        )
        # pylint: enable=protected-access

        # Lets apply observation transformations from wrappers.
        # WARNING: This assumes that only ObservationWrapper(s) in the wrappers
        # stack transform observation.
        env = self._env
        observation_wrappers = []
        while True:
            if isinstance(env, gym.ObservationWrapper):
                observation_wrappers.append(env)
            if isinstance(env, football_env.wrappers.FrameStack):
                return env._get_observation()  # pylint: disable=protected-access
            if isinstance(env, gym.Wrapper):
                env = env.env
            else:
                break

        for wrapper in reversed(observation_wrappers):
            observation = wrapper.observation(observation)

        return observation

    @property
    def state_info(self):
        observations = self._env.unwrapped.observation()
        assert len(observations) == 1
        return {
            key: value
            for (key, value) in observations[0].items()
            if key in {
                'left_team', 'left_team_direction',
                'right_team', 'right_team_direction',
                'ball', 'ball_direction', 'ball_owned_team',
                'active', 'game_mode',
            }
        }
