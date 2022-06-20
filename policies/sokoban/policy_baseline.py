import gc
import os

import numpy as np
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    Softmax,
)
from tensorflow.keras.models import (
    load_model,
    Model,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2

from envs import Sokoban
from metric_logging import log_scalar
from supervised import DataCreatorPolicyBaselineSokoban


class SokobanPolicyBaseline:
    def __init__(
        self,
        num_layers=5,
        batch_norm=True,
        model_id=None,
        learning_rate=0.01,
        kernel_size=(3, 3),
        weight_decay=0.
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

    def construct_networks(self):
        if self._model is None:
            if self.model_id is None:
                input_state = Input(batch_shape=(None, None, None, 7))
                layer = input_state

                for _ in range(self.num_layers):
                    layer = Conv2D(
                        filters=64,
                        kernel_size=self.kernel_size,
                        padding='same',
                        activation='relu',
                        kernel_regularizer=l2(self.weight_decay),
                    )(layer)

                    if self.batch_norm:
                        layer = BatchNormalization()(layer)

                layer = GlobalAveragePooling2D()(layer)
                # layer = Flatten()(layer)
                layer = Dense(4, activation=None, kernel_regularizer=l2(self.weight_decay))(layer)
                output = Softmax()(layer)

                self._model = Model(inputs=input_state, outputs=output)
                self._model.compile(
                    loss='categorical_crossentropy',
                    metrics='accuracy',
                    optimizer=Adam(learning_rate=self.learning_rate)
                )
                self.data_creator = DataCreatorPolicyBaselineSokoban()
            else:
                self.load_model(self.model_id)

    def load_data(self, dataset_file):
        self.data_creator.load(dataset_file)

    def fit_and_dump(self, x_train, y_train, x_val, y_val, epochs, dump_folder, checkpoints=None):
        for epoch in range(epochs):
            history = self._model.fit(x_train, y_train, epochs=1, validation_data=(x_val, y_val), verbose=2)
            train_history = history.history

            for metric, value in train_history.items():
                log_scalar(metric, epoch, value[0])

            if checkpoints is not None and epoch in checkpoints:
                print(f'Saving model after {epoch} epochs.')
                self.save_model(os.path.join(dump_folder, f'epoch_{epoch}'))

            # Clean accumulated data
            gc.collect()

        self.save_model(os.path.join(dump_folder, f'epoch_{epoch}'))

    def save_model(self, model_id):
        self._model.save(model_id)

    def load_model(self, model_id):
        self._model = load_model(model_id)

    def predict_actions(self, input):
        prediction = self._model.predict(np.array([input]))[0]
        return list(prediction)
