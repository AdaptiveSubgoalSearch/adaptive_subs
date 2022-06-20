import itertools
from os import listdir
from os.path import (
    isdir,
    join,
)
import pickle
import random

from joblib import load
import numpy as np
from tqdm import tqdm

from utils.utils_sokoban import (
    agent_coordinates_to_action,
    detect_dim_room,
    detect_num_boxes,
    get_field_name_from_index,
)


class DataCreatorConditionalPolicySokoban:
    def __init__(self, validation_split=None, keep_trajectories=1, max_distance=None, final_skip=None):
        self.validation_split = validation_split
        self._keep_trajectories = keep_trajectories
        self.data = {}
        self.training_keys = None
        self.validation_keys = None
        self.dim_room = None
        self.num_boxes = None
        self.max_distance = max_distance
        self.final_skip = final_skip

    def load(self, dataset_path=None):
        if isdir(dataset_path):
            files = listdir(dataset_path)

            for file in files:
                print(f'Loading data from file {file}.')
                part_dict = load(join(dataset_path, file))
                self.data.update(part_dict)
        else:
            with open(dataset_path, 'rb') as handle:
                self.data = pickle.load(handle)

        all_keys_shuffled = list(self.data.keys()).copy()
        random.shuffle(all_keys_shuffled)
        val_split_num = int(len(all_keys_shuffled) * self.validation_split) + 1
        self.validation_keys = all_keys_shuffled[:val_split_num]
        self.training_keys = all_keys_shuffled[val_split_num:]

        assert len(self.training_keys) > 0

        self.dim_room = detect_dim_room(self.data[0][0])
        self.num_boxes = detect_num_boxes(self.data[0][0])

    def create_train_and_validation_sets(self):
        """
        Returns four numpy arrays in following order: training X, training Y, validation X, validation Y. X arrays
        have shape (number of observations, board dimension, board dimension, 14). Y arrays have shape (number of
        observations, 4).
        """
        assert self.training_keys is not None and self.validation_keys is not None, 'You must load data first.'

        print('Processing training dataset.')
        x_train, y_train, _ = self.create_xy(self.training_keys)
        print('Processing validation dataset.')
        x_validation, y_validation, distances = self.create_xy(self.validation_keys)

        print(f'Train set has {len(x_train)} elements.')
        print(f'Validation set has {len(x_validation)} elements.')

        for i in range(10):
            print(f'Sample training datapoint')
            print(np.argmax(x_train[i, :, :, :7], axis=-1))
            print(np.argmax(x_train[i, :, :, 7:], axis=-1))
            print(f'action: {y_train[i]}\n')

        return x_train, y_train, x_validation, (y_validation, distances)

    def create_xy(self, keys):
        x = []
        y = []
        distances = []

        for key in tqdm(keys):
            if random.random() > self._keep_trajectories:
                continue

            num_actions = len(self.data[key]) - 1
            new_xs = []
            new_ys = []

            for idx in range(num_actions):
                if np.array_equal(self.data[key][idx], self.data[key][idx + 1]):
                    continue

                action = np.zeros(4)
                action_idx = self.detect_action(self.data[key][idx], self.data[key][idx + 1])
                action[action_idx] = 1
                new_xs.append(self.data[key][idx].copy())
                new_ys.append(action)

            for idx in range(len(new_xs) - self.final_skip):
                d = np.random.randint(1, self.max_distance + 1)
                # for d in range(1, standard_dist+1):
                if idx + d >= len(new_xs):
                    d = max(1, len(new_xs) - 1 - idx)
                    if idx + d >= len(new_xs):
                        continue
                x.append(np.concatenate([new_xs[idx].copy(), new_xs[idx + d].copy()], axis=-1))
                y.append(new_ys[idx])
                distances.append(d)

        assert len(x) == len(y) and len(y) == len(distances)

        shuffle = np.random.permutation(len(x))

        x = np.array(x)[shuffle]
        y = np.array(y)[shuffle]
        distances = np.array(distances)[shuffle]

        return x, y, distances

    def detect_action(self, board_before, board_after):
        x_before, y_before = self.get_agent_position(board_before)
        x_after, y_after = self.get_agent_position(board_after)
        delta_x = x_after - x_before
        delta_y = y_after - y_before

        return agent_coordinates_to_action(delta_x, delta_y)

    def get_agent_position(self, board):
        for xy in itertools.product(list(range(self.dim_room[0])), list(range(self.dim_room[1]))):
            x, y = xy
            object = get_field_name_from_index(np.argmax(board[x][y]))

            if object == 'agent':
                return x, y

            if object == 'agent_on_goal':
                return x, y

        assert False, 'No agent on the board'
