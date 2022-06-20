import os
import tensorflow.keras.layers as tl
import tensorflow.keras as tfk
from tensorflow.python.keras.regularizers import l1_l2
from metric_logging import log_scalar
import numpy as np

_thresholds_fp = [0.5, 0.6, 0.7, 0.8, 0.9]
_thresholds_fn = [0.1, 0.2, 0.3, 0.4, 0.5]


class LogCallback(tfk.callbacks.Callback):
    def __init__(self, train_len, valid_len):
        super().__init__()
        self.train_len = train_len
        self.valid_len = valid_len

    def on_epoch_end(self, epoch, logs=None):
        print('On_epoch_end logs', logs)
        for metric, value in logs.items():
            if not isinstance(value,(list, tuple, np.ndarray)):
                log_scalar(metric, epoch, value)
            else:
                is_val = 'val' in metric
                ts = _thresholds_fp if 'val' in metric else _thresholds_fn
                for t, v in zip(ts, value):
                    v = round(v/self.train_len*100 if not is_val else v/self.valid_len*100, 2)
                    log_scalar(f'{metric}{str(t)}', epoch, v)


def create_scheduler(drops):
    print('Creating scheduler with following drops', drops)
    return lambda epoch, lr: drops[epoch] * lr if epoch in drops else lr


class VerificatorSokobanCNN:
    def __init__(self, num_layers=5, batch_norm=True, model_id=None, lr=0.01, kernel_size=(4, 4),
     weight_decay=0., momentum=0.9, optimizer='adam', dropout_head=None, dropout_inner=None, l1=None, l2=None,
     scheduler_drops={}, conv_class='conv2d'):
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.scheduler_drops = scheduler_drops

        self.dropout_head = dropout_head
        self.dropout_inner = dropout_inner

        self.l1 = l1
        self.l2 = l2
        self.conv_class = tl.Conv2D if conv_class == 'conv2d' else tl.SeparableConv2D

        self.model_id = model_id
        self._model = None
        if optimizer == 'sgd':
            self.optimizer = tfk.optimizers.SGD(learning_rate=lr, momentum=momentum)
        elif optimizer == 'adam':
            self.optimizer = tfk.optimizers.Adam(learning_rate=lr)
        elif optimizer == 'rmsprop':
            self.optimizer = tfk.optimizers.RMSprop(lr=lr, momentum=momentum)
        else:
            raise ValueError('Unknown optimizer')
        self.kernel_size = kernel_size
        self.weight_decay = weight_decay

    @property
    def model(self) -> tfk.Model:
        return self._model

    def construct_network(self):
        if self._model is not None:
            raise Exception('Already constructed')

        if self.model_id:
            self._model = tfk.models.load_model(self.model_id)
            return

        self._construct_merge_network()

    def _construct_merge_network(self):
        input_state = tl.Input(batch_shape=(None, 12, 12, 14))
        layer = input_state

        for _ in range(self.num_layers):
            layer = self.conv_class(filters=64, kernel_size=self.kernel_size, padding='same',
                                    activation='relu', kernel_regularizer=l1_l2(l1=self.l1, l2=self.l2))(layer)
            if self.batch_norm:
                layer = tl.BatchNormalization()(layer)
            if self.dropout_inner:
                layer = tl.Dropout(self.dropout_inner)(layer)

        layer = tl.GlobalAveragePooling2D()(layer)
        if self.dropout_head:
            layer = tl.Dropout(self.dropout_head)(layer)
        output_layer = tl.Dense(1, activation='sigmoid')(layer)

        self._model = tfk.Model(inputs=input_state, outputs=output_layer)
        self._model.compile(optimizer=self.optimizer, loss='binary_crossentropy',
                            metrics=['accuracy',
                                     tfk.metrics.FalseNegatives(thresholds=_thresholds_fn),
                                     tfk.metrics.FalsePositives(thresholds=_thresholds_fp)])
        print(self._model.summary())
        print(self._model.input)
        print(self._model.output)

    def fit_and_dump(self, train_seq, val_seq, epochs, dump_folder, batch_size):
        try:
            self._model.fit(x=train_seq, epochs=epochs, validation_data=val_seq,
                            verbose=2, batch_size=batch_size, validation_batch_size=batch_size,
                            callbacks=[LogCallback(len(train_seq), len(val_seq)),
                                       tfk.callbacks.LearningRateScheduler(create_scheduler(self.scheduler_drops), verbose=1)])
        except Exception as e:
            print('Got exception', str(e))
            raise ValueError()
        self._model.save(os.path.join(dump_folder, f'epoch_{epochs}'))
