import itertools

import tensorflow
import numpy as np

from utils.utils_sokoban import get_field_name_from_index


class ValueEstimator:
    def __init__(self, model_id):
        self.value_network_id = model_id
        self._model = None

    def evaluate(self, state):
        return self._model.predict(np.array([state]))[0][0] + self.reward_for_boxes_on_goals(state)

    def construct_networks(self):
        if self._model is None:
            self._model = tensorflow.keras.models.load_model(self.value_network_id)

    def reward_for_boxes_on_goals(self, state):
        reward = 0
        for xy in itertools.product(list(range(state.shape[0])),
                                    list(range(state.shape[1]))):
            x, y = xy
            if get_field_name_from_index(np.argmax(state[x][y])) == 'box_on_goal':
                reward += 1
        return reward
