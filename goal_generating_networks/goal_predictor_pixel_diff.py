import gc
import os

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPool2D,
    Softmax,
    ZeroPadding2D,
    UpSampling2D,
    Lambda,
    AveragePooling2D,
    Activation,
    add,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2

from envs import Sokoban
from metric_logging import log_scalar
from supervised import DataCreatorSokobanPixelDiff


class GoalPredictorPixelDiff:
    def __init__(
        self,
        num_layers=5,
        batch_norm=True,
        model_id=None,
        learning_rate=0.01,
        kernel_size=(5, 5),
        weight_decay=0.,
        dropout=0.,
        batch_size=32,
        policy_for_reachability=None
    ):

        self.core_env = Sokoban()
        self.dim_room = self.core_env.get_dim_room()

        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.model_id = model_id

        self._model = None
        self._predictions_counter = 0
        self.learning_rate = learning_rate
        self.kernel_size = kernel_size
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size

        self.conditional_policy = None
        if policy_for_reachability is not None:
            self.conditional_policy = policy_for_reachability()
            self.conditional_policy.construct_networks()

    def construct_networks(self):
        if self._model is None:
            if self.model_id is None:
                print(f'Constructing goal predictor from ID: {self.model_id}')
                input_state = Input(batch_shape=(None, None, None, 7))
                input_condition = Input(batch_shape=(None, None, None, 7))

                layer = Concatenate()([input_state, input_condition])

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

                branch1 = Dense(7, activation=None, kernel_regularizer=l2(self.weight_decay))(layer)
                branch1 = Flatten()(branch1)

                branch2 = Dense(1, activation=None, kernel_regularizer=l2(self.weight_decay))(layer)
                # branch2 = Flatten()(branch2)
                branch2 = GlobalAveragePooling2D()(branch2)

                output = Concatenate()([branch1, branch2])
                output = Softmax()(output)

                self._model = Model(inputs=[input_state, input_condition], outputs=output)
                self._model.compile(
                    loss='categorical_crossentropy',
                    metrics='accuracy',
                    optimizer=Adam(learning_rate=self.learning_rate)
                )

                self.data_creator = DataCreatorSokobanPixelDiff()
            else:
                self.load_model(self.model_id)

    def reset_predictions_counter(self):
        self._predictions_counter = 0

    def read_predictions(self):
        return self._predictions_counter

    def load_data(self, dataset_file):
        self.data_creator.load(dataset_file)

    def fit_and_dump(self, x, y, validation_data, epochs, dump_folder, checkpoints=None):
        for epoch in range(epochs):
            history = self._model.fit(x, y, batch_size=self.batch_size, epochs=1,
                                      validation_data=validation_data, verbose=2)
            train_history = history.history
            for metric, value in train_history.items():
                log_scalar(metric, epoch, value[0])
            # self.log_reachability(epoch, validation_data[0])
            if checkpoints is not None and epoch in checkpoints:
                print(f'saving model after {epoch} epochs.')
                self.save_model(os.path.join(dump_folder, f'epoch_{epoch}'))
            gc.collect()
        self.save_model(os.path.join(dump_folder, f'epoch_{epoch}'))

    # def log_reachability(self, epoch, data):
    #     if self.conditional_policy is None:
    #         return
    #
    #     print('data', data.shape, data)
    #
    #     samples = np.random.choice(data, size=10, replace=False)
    #     data_x = np.array([sample[0] for sample in samples])
    #     data_y = self._model.predict(data_x)
    #
    #     score = np.mean(
    #         self.conditional_policy.can_reach_batched(data_x, data_y, [12 for _ in samples], batch_size=1000,
    #                                                   sampling='greedy'))
    #     log_scalar('subgoal reachability', epoch, score)

    def predict_pdf(self, input, condition):
        self._predictions_counter += 1
        raw = self._model.predict([np.array([input]), np.array([condition])])[0]
        return raw

    def predict_pdf_batch(self, input_boards, conditions):
        self._predictions_counter += 1
        raw = self._model.predict([input_boards, conditions])
        return raw

    def save_model(self, model_id):
        self._model.save(model_id)

    def load_model(self, model_id):
        self._model = keras.models.load_model(model_id)

    def flat_to_2d(self, n):
        element = n % 7
        base_n = n // 7
        x = base_n // self.dim_room[0]
        y = base_n % self.dim_room[1]

        return x, y, element

    def sample(self, pdf):
        out = [self.flat_to_2d(idx) for idx in reversed(np.argsort(pdf))]
        out_p = [pdf[idx] for idx in reversed(np.argsort(pdf))]
        return out, out_p

    def sample_with_internal_confidence_level(self, pdf, internal_confidence_level):
        assert internal_confidence_level > 0 and internal_confidence_level < 1, 'confidence_level must be between 0 and 1'
        out = []
        out_p = []

        for idx in reversed(np.argsort(pdf)):
            out.append(self.flat_to_2d(idx))
            out_p.append(pdf[idx])

            if sum(out_p) > internal_confidence_level:
                break

        return out, out_p
