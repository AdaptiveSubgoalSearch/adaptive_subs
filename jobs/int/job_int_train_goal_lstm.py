import json
import os
import time

from tensorflow import keras

from goal_generating_networks import int_lstm
from jobs.core import Job
from metric_logging import log_scalar
from supervised import int
from supervised.int.gen_subgoal_data import generate_data


class JobIntTrainGoalLSTM(Job):
    def __init__(
            self,
            representation=int.InfixRepresentation,
            n_iterations=1000,
            proofs_per_iteration=10,
            latent_dim=256,
            batch_size=32,
            epochs_per_iteration=10,
            learning_rate=0.0001,
    ):
        self.n_iterations = n_iterations
        self.proofs_per_iteration = proofs_per_iteration
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs_per_iteration = epochs_per_iteration
        self.learning_rate = learning_rate
        self.representation = representation

    def execute(self):
        combo_path = './assets/int/benchmark/field/'
        kl_dict = json.load(open(os.path.join(combo_path, "orders.json"), "r"))

        token_consts = self.representation.token_consts
        model = int_lstm.lstm_goal_predictor(
            latent_dim=self.latent_dim,
            token_consts=token_consts
        )
        opt = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        model.compile(
            optimizer=opt, loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                int_lstm.get_function_accuracy_ignore_padding(
                    token_consts.padding_token
                ),
                int_lstm.perfect_sequence
            ]
        )

        done_epochs = 0
        for iteration in range(self.n_iterations):
            encoder_input_data, decoder_input_data, decoder_target_data = (
                generate_data(self.proofs_per_iteration, kl_dict, self.representation)
            )

            t = time.time()
            history = model.fit(
                [encoder_input_data, decoder_input_data],
                decoder_target_data,
                batch_size=self.batch_size,
                validation_split=0.2,
                initial_epoch=done_epochs,
                epochs=done_epochs + self.epochs_per_iteration,
            )
            done_epochs += self.epochs_per_iteration
            for metric_name, values in history.history.items():
                for epoch, value in zip(history.epoch, values):
                    log_scalar(metric_name, epoch, value)
            # Save model
            print("iteration training took", time.time() - t)
            model.save(f"out/sequence_model_{iteration}")
