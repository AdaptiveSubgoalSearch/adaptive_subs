import os
from metric_logging import log_scalar
import tensorflow


from tensorflow.keras.layers import Input, Conv2D, Dense, BatchNormalization, MaxPool2D, Softmax, Concatenate, Flatten
from tensorflow.keras.models import Model

from envs import Sokoban
from supervised.data_creator_sokoban import DataCreatorSokoban

import numpy as np


class ConditionalGoalPredictorSokoban:
    def __init__(self, num_layers=5, batch_norm=True, model_id=None):

        self.core_env = Sokoban()
        self.dim_room = self.core_env.get_dim_room()

        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.model_id = model_id

        self._model = None
        self._predictions_counter = 0

    def construct_networks(self):
        if self._model is None:
            if self.model_id is None:
                input_state = Input(batch_shape=(None, None, None, 7))
                input_condition = Input(batch_shape=(None, None, None, 7))

                layer = Concatenate()([input_state, input_condition])

                for _ in range(self.num_layers):
                    layer = Conv2D(filters=64, kernel_size=(5, 5),
                                   padding='same', activation='relu')(layer)
                    if self.batch_norm:
                        layer = BatchNormalization()(layer)
                layer = Dense(1, activation='relu')(layer)
                layer = Flatten()(layer)
                output_layer = Softmax()(layer)

                self._model = Model(inputs=[input_state, input_condition], outputs=output_layer)
                self._model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

                self.data_creator = DataCreatorSokoban()
            else:
                self.load_model(self.model_id)

    def reset_predictions_counter(self):
        self._predictions_counter = 0

    def read_predictions(self):
        return self._predictions_counter

    def load_data(self, dataset_file):
        self.data_creator.load(dataset_file)

    def fit_and_dump(self, x, y, validation_data, epochs, dump_folder,
                     checkpoints=None):

        # try:
        #     print(f'making directory {dump_folder}')
        #     os.mkdir(dump_folder)
        # except:
        #     print(f'cannot make directory {dump_folder}')

        for epoch in range(epochs):
            history = self._model.fit(x, y, epochs=1, validation_data=validation_data, verbose=2)
            train_history = history.history
            for metric, value in train_history.items():
                log_scalar(metric, epoch, value[0])
            if checkpoints is not None and epoch in checkpoints:
                print(f'saving model after {epoch} epochs.')
                self.save_model(os.path.join(dump_folder, f'epoch_{epoch}'))
        self.save_model(os.path.join(dump_folder, f'epoch_{epoch}'))

    def predict_pdf(self, input, condition):
        self._predictions_counter += 1
        raw =  self._model.predict([np.array([input]), np.array([condition])])[0]
        return raw

    def predict_pdf_batch(self, input_boards, conditions):
        self._predictions_counter += 1
        raw =  self._model.predict([input_boards, conditions])
        return raw

    def save_model(self, model_id):
        self._model.save(model_id)

    def load_model(self, model_id):
        self._model = tensorflow.keras.models.load_model(model_id)

    def flat_to_2d(self, n):
        x = n // self.dim_room[0]
        y = n % self.dim_room[1]
        return x, y

    def smart_sample(self, pdf, internal_confidence_level):
        assert internal_confidence_level > 0 and internal_confidence_level < 1, 'confidence_level must be between 0 and 1'
        out = []
        out_p = []
        for idx in reversed(np.argsort(pdf)):
            out.append(self.flat_to_2d(idx))
            out_p.append(pdf[idx])
            if sum(out_p) > internal_confidence_level:
                break
        return out, out_p
