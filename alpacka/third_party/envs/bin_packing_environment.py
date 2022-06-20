"""Source code copied and adjusted from or-rl-benchmarks.

GitHub:
https://github.com/awslabs/or-rl-benchmarks

License:
Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import gym
import numpy as np


BIG_NEG_REWARD = -100
BIG_POS_REWARD = 10


class BinPackingGymEnvironment(gym.Env):
    """ Bin Packing Problem as RL environment



    STATE:
        Number of bags at each level
        Item size
    ACTION:
        Choose bag
    """

    def __init__(
        self, bag_capacity=9, item_sizes=(2, 3), item_probabilities=(0.8, 0.2),
        time_horizon=1000,
    ):
        self.bag_capacity = bag_capacity
        self.item_sizes = item_sizes
        self.item_probabilities = item_probabilities
        self.time_horizon = time_horizon
        self.num_bins = bag_capacity

        print('Using bin size: ', self.bag_capacity)
        print(f'Using items sizes {self.item_sizes}\n'
              f'With item probabilities {self.item_probabilities}')

        self.episode_count = 0

        # state: number of bags at each level, item size,
        self.observation_space = gym.spaces.Box(
            low=np.array([0] * self.num_bins + [0]),
            high=np.array(
                [self.time_horizon] * self.num_bins + [max(self.item_sizes)]
            ),
            dtype=np.uint32
        )

        # actions: select a bag from the different levels possible
        self.action_space = gym.spaces.Discrete(self.bag_capacity)

        self.time_remaining = self.time_horizon
        self.item_size = None
        self.num_full_bags = 0
        self.num_bins_levels = [0] * self.num_bins
        self.total_reward = 0
        self.waste = 0
        self.bin_type_distribution_map = {}
        self._random_choices = None

    def reset(self):
        self.time_remaining = self.time_horizon
        self.num_full_bags = 0
        self.item_size = self._get_item()
        # an array of size bag capacity that keeps track of
        # number of bags at each level
        self.num_bins_levels = [0] * self.num_bins

        initial_state = self.num_bins_levels + [self.item_size]
        self.total_reward = 0
        self.waste = 0
        self.episode_count += 1
        # level to bin types, to the # of bins for each bin type.
        self.bin_type_distribution_map = {}

        return initial_state

    def step(self, action):
        done = False
        if action >= self.bag_capacity:  # pylint: disable=no-else-raise
            raise ValueError('Error: Invalid Action')
        elif action > (self.bag_capacity - self.item_size):
            # can't insert item because bin overflow
            reward = BIG_NEG_REWARD - self.waste
            done = True
        elif action == 0:  # new bag
            self.num_bins_levels[self.item_size] += 1
            # waste = sum of empty spaces in all bags
            self.waste = self.bag_capacity - self.item_size
            # reward is negative waste
            reward = -1 * self.waste
            self._update_bin_type_distribution_map(0)
        elif self.num_bins_levels[action] == 0:
            # can't insert item because bin of this level doesn't exist
            reward = BIG_NEG_REWARD - self.waste
            done = True
        else:
            if action + self.item_size == self.bag_capacity:
                self.num_full_bags += 1
            else:
                self.num_bins_levels[action + self.item_size] += 1
            # waste = empty space in the bag
            self.waste = -1 * self.item_size
            # reward is negative waste
            reward = -1 * self.waste
            self._update_bin_type_distribution_map(action)
            if self.num_bins_levels[action] < 0:
                print(self.num_bins_levels[action])
            self.num_bins_levels[action] -= 1

        self.total_reward += reward

        self.time_remaining -= 1
        if self.time_remaining == 0:
            done = True

        # get the next item
        self.item_size = self._get_item()
        # state is the number of bins at each level and the item size
        state = self.num_bins_levels + [self.item_size]
        info = self.bin_type_distribution_map

        return state, reward, done, info

    def _get_item(self):
        num_items = len(self.item_sizes)
        item_index = np.random.choice(num_items, p=self.item_probabilities)
        return self.item_sizes[item_index]

    def _update_bin_type_distribution_map(self, target_bin_util):
        if (
            target_bin_util < 0 or
            target_bin_util + self.item_size > self.bag_capacity
        ):
            # print('Error: Invalid Bin Utilization/Item Size')
            return
        elif (
            target_bin_util > 0 and
            target_bin_util not in self.bin_type_distribution_map
        ):
            # print('Error: bin_type_distribution_map does not '
            #       'contain ' + str(target_bin_util) + ' as key!')
            return
        elif (
            target_bin_util > 0 and
            target_bin_util in self.bin_type_distribution_map and
            len(self.bin_type_distribution_map[target_bin_util]) == 0
        ):
            # print('Error: bin_type_distribution_map has no element at '
            #       'level ' + str(target_bin_util) + ' !')
            return
        elif target_bin_util == 0:  # opening a new bin
            if self.item_size not in self.bin_type_distribution_map:
                self.bin_type_distribution_map[self.item_size] = \
                    {str(self.item_size): 1}
            elif (
                str(self.item_size) not in
                self.bin_type_distribution_map[self.item_size]
            ):
                self.bin_type_distribution_map[
                    self.item_size][str(self.item_size)] = 1
            else:
                self.bin_type_distribution_map[
                    self.item_size][str(self.item_size)] += 1
        else:
            key = np.random.choice(
                list(self.bin_type_distribution_map[target_bin_util].keys()))
            if self.bin_type_distribution_map[target_bin_util][key] <= 0:
                # print('Error: Invalid bin count!')
                return
            elif self.bin_type_distribution_map[target_bin_util][key] == 1:
                del self.bin_type_distribution_map[target_bin_util][key]
            else:
                self.bin_type_distribution_map[target_bin_util][key] -= 1

            new_key = self._update_key_for_bin_type_distribution_map(
                key, self.item_size)
            if (
                (target_bin_util + self.item_size) not in
                self.bin_type_distribution_map
            ):
                self.bin_type_distribution_map[
                    target_bin_util + self.item_size] = {new_key: 1}
            elif (
                new_key not in
                self.bin_type_distribution_map[target_bin_util + self.item_size]
            ):
                self.bin_type_distribution_map[
                    target_bin_util + self.item_size][new_key] = 1
            else:
                self.bin_type_distribution_map[
                    target_bin_util + self.item_size][new_key] += 1

    @staticmethod
    def _update_key_for_bin_type_distribution_map(key, item_size):
        parts = key.split(' ')
        parts.append(str(item_size))
        parts.sort()
        return ' '.join(parts)

    def render(self, mode='human'):
        pass


class BinPackingNewBinForInvalidAction(BinPackingGymEnvironment):
    """

    This is a copy of BinPackingNearActionGymEnvironment from https://github.com/awslabs/or-rl-benchmarks/blob/master/Bin%20Packing/src/bin_packing_environment.py  # pylint: disable=line-too-long
    but instead of the nearest valid action we create a new bucket.
    """
    def step(self, action):
        done = False
        invalid_action = not self.is_action_valid(action)
        if invalid_action:
            action = 0
            invalid_action = True

        reward = self._insert_item(action)

        self.total_reward += reward

        self.time_remaining -= 1
        if self.time_remaining == 0:
            done = True

        # get the next item
        self.item_size = self._get_item()
        # state is the number of bins at each level and the item size
        state = self.num_bins_levels + [self.item_size]
        info = self.bin_type_distribution_map
        info['invalid_action'] = invalid_action
        return state, reward, done, info

    def _insert_item(self, action):
        if action == 0:  # new bag
            self.num_bins_levels[self.item_size] += 1
            # waste added by putting in new item
            self.waste = self.bag_capacity - self.item_size
        else:  # insert in existing bag
            if action + self.item_size == self.bag_capacity:
                self.num_full_bags += 1
            else:
                self.num_bins_levels[action + self.item_size] += 1
            self.num_bins_levels[action] -= 1
            # waste reduces as we insert item in existing bag
            self.waste = -1 * self.item_size
        # reward is negative waste
        reward = -1 * self.waste
        self._update_bin_type_distribution_map(action)
        return reward

    def is_action_valid(self, action):
        """Return true if action is valid in self.state"""
        if action >= self.bag_capacity:  # pylint: disable=no-else-raise
            raise ValueError('Error: Invalid Action')
        elif action > (self.bag_capacity - self.item_size):
            # can't insert item because bin overflow
            return False
        elif action == 0:  # new bag
            return True
        elif self.num_bins_levels[action] == 0:
            return False
        else:  # insert in existing bag
            return True
