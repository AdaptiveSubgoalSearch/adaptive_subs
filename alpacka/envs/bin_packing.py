"""Bin Packing environment."""

import copy

import gym
import numpy as np

from alpacka.envs import base
from alpacka.third_party.envs import bin_packing_environment


class BinPacking(base.RestorableEnv):
    """Bin Packing Problem as RL environment"""

    stochasticity = base.Stochasticity.universal

    class BinPackingState:
        """Hashable state of BinPacking"""
        hash_key = np.random.RandomState(0).normal(size=10000)  # pylint: disable=no-member

        def __init__(self,
                     num_bins_levels,
                     item_size,
                     time_remaining,
                     time_remaining_equality_ignore=False):
            self.num_bins_levels = num_bins_levels
            self.item_size = item_size
            self.time_remaining = time_remaining
            self._hash = None
            self._time_remaining_equality_ignore = (
                time_remaining_equality_ignore)

        # pylint: disable=redefined-builtin
        def __hash__(self):
            if self._hash is None:
                num_bins_levels = np.asarray(self.num_bins_levels)
                hash = np.dot(num_bins_levels,
                              self.hash_key[:len(num_bins_levels)])
                hash += self.item_size * self.hash_key[-1]
                if not self._time_remaining_equality_ignore:
                    hash += self.time_remaining * self.hash_key[-1]

                self._hash = int(hash * 10e8)
            return self._hash

        def __eq__(self, other):
            if self.item_size != other.item_size:
                return False
            if (
                self.time_remaining != other.time_remaining and
                not self._time_remaining_equality_ignore
            ):
                return False
            return self.num_bins_levels == other.num_bins_levels

    def __init__(
        self,
        bag_capacity=9,
        item_sizes=(2, 3),
        item_probabilities=(0.8, 0.2),
        time_horizon=1000,
        reward_scale=1.,
        bin_observation_scale=1.,
        time_in_observation=False,
        time_remaining_equality_ignore=False,
    ):
        """Initializes BinPacking

        Args:
            bag_capacity (int): limit of total size of items in the single bag.
            item_sizes (tuple): possible sizes of items.
            item_probabilities (tuple): distribution over item sizes sampled
                at each timestep.
            time_horizon (int): number of timesteps in singe episode.
            reward_scale (float): rescaling factor for rewards
            bin_observation_scale (float): rescaling factor for part of
                observation encoding number of bags with given filling level.
            time_in_observation (bool): if observation should consist scalar
                (in range [0,1]) encoding remaining number of timesteps.
            time_remaining_equality_ignore: if time remaining should be taken
                into consideration while comparing two states (this works only
                when time_in_observation is set to True)
        """
        self._env = bin_packing_environment.BinPackingNewBinForInvalidAction(
            bag_capacity=bag_capacity,
            item_sizes=item_sizes,
            item_probabilities=item_probabilities,
            time_horizon=time_horizon,
        )
        self._reward_scale = reward_scale
        self._bin_observation_scale = bin_observation_scale
        self.time_in_obs = time_in_observation
        self._time_remaining_equality_ignore = time_remaining_equality_ignore

        # self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        additional_obs_dim = 1 if self.time_in_obs else 0
        self.observation_space = gym.spaces.Box(
            shape=(self._env.observation_space.shape[0] + additional_obs_dim,),
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
        )

    def _get_observation(self):
        num_bins = self._env.num_bins

        obs = self._env.num_bins_levels + [self._env.item_size]
        if self.time_in_obs:
            obs += [self._env.time_remaining / self._env.time_horizon]
        obs = np.array(obs, dtype=np.float32)
        obs[:num_bins] = (
            obs[:num_bins] * self._bin_observation_scale
        )

        return obs

    def step(self, action):
        _, reward, done, info = self._env.step(action)
        reward *= self._reward_scale
        return self._get_observation(), reward, done, info

    def reset(self):
        self._env.reset()
        return self._get_observation()

    def clone_state(self):
        """Returns the current environment state."""
        return self.BinPackingState(
            num_bins_levels=self._env.num_bins_levels.copy(),
            item_size=self._env.item_size,
            time_remaining=self._env.time_remaining,
            time_remaining_equality_ignore=self._time_remaining_equality_ignore,
        )

    def restore_state(self, state):
        """Restores environment state, returns the observation."""
        # WARNING: restoring state does not update several attributes of
        # underlying environment (self._env).
        # These attributes are:
        #   bin_type_distribution_map
        #   num_full_bags
        #   waste
        #   total_reward
        #   episode_count
        assert isinstance(state, self.BinPackingState)
        state = copy.deepcopy(state)
        self._env.num_bins_levels = state.num_bins_levels
        self._env.item_size = state.item_size
        self._env.time_remaining = state.time_remaining
        return self._get_observation()

    def render(self, mode='human'):
        pass
