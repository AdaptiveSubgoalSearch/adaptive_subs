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


class DataCreatorPolicyBaselineSokoban:
    def __init__(self, validation_split=None, keep_trajectories=1):
        self.validation_split = validation_split
        self._keep_trajectories = keep_trajectories
        self.data = {}
        self.training_keys = None
        self.validation_keys = None
        self.dim_room = None
        self.num_boxes = None

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
        Returns four numpy arrays in following order: training X, traaining Y, validation X, validation Y. X arrays
        have shape (number of observations, board dimension, board dimension, 7). Y arrays have shape (number of
        observations, 4).
        """
        assert self.training_keys is not None and self.validation_keys is not None, 'You must load data first.'

        print('Processing training dataset.')
        x_train, y_train = self.create_xy(self.training_keys)
        print('Processing validation dataset.')
        x_validation, y_validation = self.create_xy(self.validation_keys)

        print(f'Train set has {len(x_train)} elements.')
        print(f'Validation set has {len(x_validation)} elements.')

        return x_train, y_train, x_validation, y_validation

    def create_xy(self, keys):
        x = []
        y = []

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

            x.extend(new_xs)
            y.extend(new_ys)

        assert len(x) == len(y)

        x = np.array(x)
        y = np.array(y)

        return x, y

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
