"""

Based on https://keras.io/examples/nlp/lstm_seq2seq/
"""
import json
import os

from jobs.core import Job
from goal_generating_networks import int_lstm
from supervised import int
from supervised.int import gen_subgoal_data


class JobIntSampleFromLSTM(Job):
    def __init__(self,
                 representation=int.InfixRepresentation,
                 checkpoint_path='out/sequence_model_0',
                 n_formulas=50,
                 length_to_pad=None):
        self.representation = representation
        self.checkpoint_path = checkpoint_path
        self.n_formulas = n_formulas
        self.length_to_pad=length_to_pad

    def execute(self):
        token_consts = self.representation.token_consts

        combo_path = './assets/int/benchmark/field/'
        kl_dict = json.load(open(os.path.join(combo_path, 'orders.json'), 'r'))

        encoder_model, decoder_model = int_lstm.load_encoder_decoder_models(
            self.checkpoint_path, token_consts
        )

        for i in range(self.n_formulas):
            encoder_input_data, x_formula, y_formula = (
                self.prepare_problem_data(kl_dict)
            )

            predicted_tokens = int_lstm.sample_output_sequence(
                encoder_input_data,
                encoder_model=encoder_model,
                decoder_model=decoder_model,
                token_consts=token_consts
            )
            decoded_formula = self.representation.formula_from_tokens(
                predicted_tokens
            )

            print(i)
            print(f'{"Input formula":>15}: {x_formula}')
            print(f'{"Decoded formula":>15}: {decoded_formula}')
            print(f'{"Target formula":>15}: {y_formula}')

    def prepare_problem_data(self, kl_dict):
        token_consts = self.representation.token_consts

        # We process a single formula at a time, so plural forms refer
        # mostly to single-element lists.
        state_pairs = gen_subgoal_data.generate_state_pairs(
            n_formulas=1, kl_dict=kl_dict
        )

        formula_pairs = gen_subgoal_data.stringify_state_data(
            state_pairs, representation=self.representation
        )
        [(x_formula, y_formula)] = formula_pairs

        [(x_tokenized, y_tokenized)] = (
            gen_subgoal_data.tokenize_stringified_data(
                formula_pairs, representation=self.representation
            )
        )
        if self.length_to_pad:
            x_tokenized += [
                token_consts.padding_token
                for _ in range(self.length_to_pad - len(x_tokenized))
            ]

        encoder_input_data, _, _ = (
            gen_subgoal_data.encode_tokenized_data_to_seq2seq_arrays(
                [(x_tokenized, y_tokenized)], token_consts=token_consts
            )
        )

        return encoder_input_data, x_formula, y_formula
