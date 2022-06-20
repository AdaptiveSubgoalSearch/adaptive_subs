import json
import os
import time

from tensorflow import keras

from goal_generating_networks import int_lstm
from jobs.core import Job
from metric_logging import log_scalar
from supervised import int
from supervised.int.gen_policy_data import generate_policy_data
from supervised.int.gen_subgoal_data import generate_data


class JobIntTrainPolicyLSTM(Job):
    def __init__(
            self,
            n_iterations=2000,
            n_formulas=10,
            max_steps_into_future=3,
            n_samples_per_proof=12,
            latent_dim=512,
            epochs_per_iteration=10,
            learning_rate=0.0001,
            representation=int.ActionRepresentationMask
    ):
        self.n_iterations = n_iterations
        self.n_formulas = n_formulas
        self.max_steps_into_future = max_steps_into_future
        self.latent_dim = latent_dim
        self.epochs_per_iteration = epochs_per_iteration
        self.learning_rate = learning_rate
        self.representation = representation

    def execute(self):
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
                generate_policy_data(self.n_formulas, self.max_steps_into_future, self.n_samples_per_proof)
            )

            t = time.time()
            history = model.fit(
                [encoder_input_data, decoder_input_data],
                decoder_target_data,
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
            model.save(f"out/policy_model_{iteration}")
