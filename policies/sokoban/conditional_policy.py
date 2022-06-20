import gc
import os

import numpy as np
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Softmax,
)
from tensorflow.keras.models import (
    load_model,
    Model,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.callbacks import LearningRateScheduler

from envs import Sokoban
from metric_logging import log_scalar
from supervised import DataCreatorConditionalPolicySokoban
from utils.utils_sokoban import create_scheduler


class SokobanConditionalPolicy:
    def __init__(
        self,
        num_layers=5,
        batch_norm=True,
        model_id=None,
        learning_rate=0.01,
        lr_drops=None,
        kernel_size=(3, 3),
        weight_decay=0.,
        dropout=0.,
        playtest_steps=None,
        playtest_samples=None,
    ):
        self.core_env = Sokoban()
        self.dim_room = self.core_env.get_dim_room()

        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.model_id = model_id

        self._model = None
        self._predictions_counter = 0

        self.learning_rate = learning_rate
        self.lr_drops = {} if lr_drops is None else lr_drops
        self.lr_scheduler = LearningRateScheduler(create_scheduler(self.lr_drops), verbose=1)
        self.optimizer = Adam(learning_rate=self.learning_rate)

        self.kernel_size = kernel_size
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.playtest_steps = playtest_steps
        self.playtest_samples = playtest_samples

    def construct_networks(self):
        if self._model is None:
            if self.model_id is None:
                input_state = Input(batch_shape=(None, None, None, 14))
                layer = input_state

                for _ in range(self.num_layers):
                    layer = Conv2D(
                        filters=64,
                        kernel_size=self.kernel_size,
                        padding='same',
                        activation='relu',
                        kernel_regularizer=l2(self.weight_decay),
                    )(layer)

                    layer = Dropout(self.dropout)(layer)

                    if self.batch_norm:
                        layer = BatchNormalization()(layer)

                layer = GlobalAveragePooling2D()(layer)
                # layer = Flatten()(layer)
                layer = Dense(4, activation=None, kernel_regularizer=l2(self.weight_decay))(layer)
                output = Softmax()(layer)
                # output = layer

                self._model = Model(inputs=input_state, outputs=output)
                self._model.compile(
                    loss='categorical_crossentropy',
                    metrics='accuracy',
                    optimizer=self.optimizer
                )
                self.data_creator = DataCreatorConditionalPolicySokoban()
            else:
                self.load_model(self.model_id)

    def load_data(self, dataset_file):
        self.data_creator.load(dataset_file)

    def fit_and_dump(self, data_creator, epochs, dump_folder, checkpoints=None):
        x_train, y_train, x_val, y_val = data_creator.create_train_and_validation_sets()
        y_val, distances = y_val
        x_playtest = self.prepare_playtest_data(x_val, distances)

        for epoch in range(epochs):
            if epoch % 10 == 0 and epoch > 0:
                x_train, y_train, x_val, y_val = data_creator.create_train_and_validation_sets()
                y_val, distances = y_val

            history = self._model.fit(x_train, y_train, initial_epoch=epoch, epochs=epoch + 1,
                                      validation_data=(x_val, y_val), verbose=2, callbacks=[self.lr_scheduler])
            train_history = history.history

            for metric, value in train_history.items():
                log_scalar(metric, epoch, value[0])
            log_scalar('learning rate', epoch, self._model.optimizer.lr.numpy())

            if checkpoints is not None and epoch in checkpoints:
                print(f'Saving model after {epoch} epochs.')
                self.save_model(os.path.join(dump_folder, f'epoch_{epoch}'))

            if epoch % 10 == 0:
                for steps in self.playtest_steps:
                    playtest_samples = np.random.default_rng().choice(len(x_playtest[steps]),
                                                                      size=self.playtest_samples, replace=False)
                    self.playtest_policy(x_playtest[steps][playtest_samples], steps, epoch)

            # Clean accumulated data
            gc.collect()

        self.save_model(os.path.join(dump_folder, f'epoch_{epoch}'))

    def can_reach(self, state, subgoal, steps):
        self.core_env.restore_full_state_from_np_array_version(state)

        for step in range(steps):
            action = np.argmax(self.predict_actions(np.concatenate([state, subgoal], axis=-1)))
            state, _, _, _ = self.core_env.step(action)

            if np.array_equal(state, subgoal):
                return True

        return False

    def can_reach_batched(self, states, subgoals, steps, batch_size=32, sampling='greedy'):
        assert len(states) == len(subgoals) and len(states) == len(steps)
        global_reached = []

        # Sort by max number of steps to batch similar queries
        queries = sorted([(step, i, state, subgoal) for i, state, subgoal, step in
                          zip(np.arange(len(states)), states, subgoals, steps)])
        assert len(queries) == len(states)

        states = [query[2] for query in queries]
        subgoals = [query[3] for query in queries]
        steps = [query[0] for query in queries]
        idx = [query[1] for query in queries]

        while not len(states) == 0:
            states_batch = states[:batch_size]
            states = states[batch_size:]

            subgoals_batch = subgoals[:batch_size]
            subgoals = subgoals[batch_size:]

            steps_batch = steps[:batch_size]
            steps = steps[batch_size:]

            # NOTE Do NOT use states, subgoals and steps henceforth, use batch variables instead.

            envs = [Sokoban() for _ in states_batch]
            for env, state in zip(envs, states_batch):
                env.restore_full_state_from_np_array_version(state)

            reached = [np.array_equal(state, subgoal) for state, subgoal in zip(states_batch, subgoals_batch)]
            ended = [step == 0 for step in steps_batch]
            current_step = 0

            while not np.all(ended):
                current_step += 1

                active_idx = np.where(np.array(ended) == False)[0]
                batch = np.array([np.concatenate([states_batch[i], subgoals_batch[i]], axis=-1) for i in active_idx])
                action_probs = self.predict_action_batch(batch)

                if sampling == 'greedy':
                    actions = np.argmax(action_probs, axis=-1)
                elif sampling == 'sampling':
                    actions = np.array([np.random.choice(4, p=probs) for probs in action_probs])
                elif sampling == 'secondary':  # take the worst action better than 0.2
                    action_probs[action_probs < 0.2] = 1.  # drop actions better than 0.2
                    actions = np.argmin(action_probs, axis=-1)  # take the worst action available
                else:
                    raise ValueError(f'Invalid sampling argument: {sampling}')

                for i, action in zip(active_idx, actions):
                    states_batch[i], _, _, _ = envs[i].step(action)
                    reached[i] = np.array_equal(states_batch[i], subgoals_batch[i])
                    ended[i] = current_step >= steps_batch[i] or reached[i]

            global_reached += reached

        result = [None] * len(global_reached)
        for i in range(len(global_reached)):
            result[idx[i]] = global_reached[i]
        return result

    def playtest_policy(self, data_x, steps, epoch):
        states = []
        subgoals = []

        for initial_state in data_x:
            state = initial_state[:,:,:7]
            subgoal = initial_state[:,:,7:]

            states.append(state)
            subgoals.append(subgoal)

        solved = sum(self.can_reach_batched(states, subgoals, [steps + 2] * len(states)))

        log_scalar(f'dist {steps} playtest', epoch, solved / len(data_x))

    def prepare_playtest_data(self, x_val, distances):
        datasets = [[] for _ in range(max(np.max(self.playtest_steps), np.max(distances)) + 1)]

        for x, dist in zip(x_val, distances):
            datasets[dist].append(x)

        print('Number of samples in distance validation datasets:')
        for i, data in enumerate(datasets):
            print(i, len(data))

        final_datasets = {d: np.array(datasets[d]) for d in self.playtest_steps}

        return final_datasets

    def save_model(self, model_id):
        self._model.save(model_id)

    def load_model(self, model_id):
        self._model = load_model(model_id)

    def predict_actions(self, input):
        prediction = self._model.predict(np.array([input]))[0]
        return list(prediction)

    def predict_action_batch(self, input):
        prediction = self._model.predict(input)
        return prediction
