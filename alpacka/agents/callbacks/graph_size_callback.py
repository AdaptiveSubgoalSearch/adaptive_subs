"""Agent callback calculating size of search graph"""

import random

import numpy as np

from alpacka import agents
from alpacka import metric_logging


class GraphSizeCallback(agents.AgentCallback):
    """Log number of distinct observations encountered during search."""

    class _HashableNumpyArray:
        hash_key = np.random.normal(size=1000000)

        def __init__(self, np_array):
            assert isinstance(np_array, np.ndarray), \
                'This works only for np.array'
            assert np_array.size <= self.hash_key.size, \
                f'Expected array of size lower than {self.hash_key.size} ' \
                f'consider increasing size of hash_key.'
            self.np_array = np_array
            self._hash = None

        def __hash__(self):
            if self._hash is None:
                flat_np = self.np_array.flatten()
                self._hash = int(np.dot(
                    flat_np,
                    self.hash_key[:len(flat_np)]) * 10e8)
            return self._hash

        def __eq__(self, other):
            return np.array_equal(self.np_array, other.np_array)

        def __ne__(self, other):
            return not self.__eq__(other)

    def __init__(self, agent, sample_rate=1.):
        """Initializes GraphSizeCallback.

        Args:
            agent (Agent): Agent which owns this callback.
            sample_rate (float): Fraction of episodes to log.
        """
        super().__init__(agent)
        self.sample_rate = sample_rate

        self._episode_observations = None
        self._step_observations = None
        self._log_current_episode = None
        self._epoch = None

    def on_episode_begin(self, env, observation, epoch):
        """Called in the beginning of a new episode."""
        del env
        self._log_current_episode = random.random() < self.sample_rate
        if not self._log_current_episode:
            return
        self._epoch = epoch
        observation = self._preprocess_observation(observation)
        self._episode_observations = {observation}
        self._step_observations = {observation}

    def on_episode_end(self):
        """Called in the end of an episode."""
        if not self._log_current_episode:
            return
        # WARNING: self.epoch is the same for many steps/episodes, this might
        # need rewriting in the future.
        metric_logging.log_scalar('episode/graph_size', self._epoch,
                                  len(self._episode_observations))

    def on_real_step(self, agent_info, action, observation, reward, done):
        """Called after every step in the real environment."""
        if not self._log_current_episode:
            return
        # WARNING: self.epoch is the same for many steps/episodes, this might
        # need rewriting in the future.
        metric_logging.log_scalar('step/graph_size', self._epoch,
                                  len(self._step_observations))
        observation = self._preprocess_observation(observation)
        self._episode_observations.add(observation)
        self._step_observations = {observation}

    def on_model_step(self, agent_info, action, observation, reward, done):
        """Called after every step in the model."""
        del agent_info, action, reward, done
        if not self._log_current_episode:
            return
        observation = self._preprocess_observation(observation)
        self._episode_observations.add(observation)
        self._step_observations.add(observation)

    def _preprocess_observation(self, observation):
        if isinstance(observation, np.ndarray):
            observation = self._HashableNumpyArray(observation)
        else:
            if not hasattr(observation, '__hash__'):
                raise ValueError('Observation should be numpy array or '
                                 'implement hashing')
        return observation
