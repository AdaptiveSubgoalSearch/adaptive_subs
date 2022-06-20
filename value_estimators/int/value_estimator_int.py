from math import exp

import torch

import transformers
from transformers import MBartForConditionalGeneration

from supervised.int import hf_data
from supervised.int import utils as int_utils
from supervised.int.gen_subgoal_data import generate_problems
from supervised.int.hf_data import GoalDataset
from supervised.int.representation.infix_value import DISTANCE_TOKENS, PADDING_LEXEME
from utils import hf
from utils import hf_generate
from visualization.seq_parse import logic_statement_to_seq_string


class TrivialValueEstimatorINT:
    def evaluate(self, states):
        return [
            -len(logic_statement_to_seq_string(
                int_utils.get_objective(state)
            ))
            for state in states
        ]

    def construct_networks(self):
        pass


class ValueEstimatorINT:
    def __init__(self, checkpoint_path=None, device=None):
        self.checkpoint_path = checkpoint_path
        self.device = device or hf.choose_device()
        self.tokenizer = hf_data.IntValueTokenizer(
            model_max_length=512,
            padding_side='right',
            pad_token=PADDING_LEXEME,
        )
        self.model = None
        self.prediction_counter = 0
        self.calls_value = 0

    def reset_counter(self):
        self.prediction_counter = 0

    def read_counter(self):
        return {'value': self.prediction_counter}

    def construct_networks(self):
        self.model = MBartForConditionalGeneration.from_pretrained(
            self.checkpoint_path
        ).to(self.device)

    def evaluate(self, states):
        return self.predict_values([
            logic_statement_to_seq_string(
                int_utils.get_objective(state)
            )
            for state in states
        ])

    @staticmethod
    def _token_scores_to_distance(scores):
        distance_scores = {}
        expected_distance = 0
        distance_probabilities = {}
        for dist, token_id in DISTANCE_TOKENS.items():
            distance_scores[dist] = scores[token_id]

        softmax_sum = sum([exp(x) for x in distance_scores.values()])
        for dist, score in distance_scores.items():
            dist_p = exp(score)/softmax_sum
            distance_probabilities[dist] = dist_p
            expected_distance += dist * dist_p

        return expected_distance

    def predict_values(self, state_strs):
        self.calls_value += 1
        self.prediction_counter += 1
        dataset = GoalDataset.from_policy_input(
            state_strs, self.tokenizer, padding=True, max_length=512
        )
        inputs = [
            hf_generate.GenerationInput(
                input_ids=entry['input_ids'],
                attention_mask=entry['attention_mask']
            )
            for entry in dataset
        ]

        model_outputs = self.model.generate(
            input_ids=torch.tensor(
                [input.input_ids for input in inputs],
                dtype=torch.int64,
                device=self.model.device,
            ),
            attention_mask=torch.tensor(
                [input.attention_mask for input in inputs],
                dtype=torch.int64,
                device=self.model.device,
            ),
            decoder_start_token_id=2,  # eos_token_id

            # This setting enforces 4 characters to be generated - desireably:
            # '$@x$' - where x stands for distance-estimating token.
            min_length=3,  # Supposedly excluding initial $
            max_length=4,  # Supposedly including initial $

            num_beams=1,
            num_return_sequences=1,
            num_beam_groups=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True
        )
        assert isinstance(
            model_outputs,
            transformers.generation_utils.GreedySearchEncoderDecoderOutput
        )
        # Index 1 means distance-estimating token (located in generated
        # seuqence at index 2), because there are no scores for initial $.
        scores_vectors = model_outputs.scores[1].cpu().numpy()
        return [
            -self._token_scores_to_distance(scores)
            for scores in scores_vectors
        ]
