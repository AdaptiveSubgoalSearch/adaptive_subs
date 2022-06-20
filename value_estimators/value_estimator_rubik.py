from math import exp

import torch

from transformers import MBartForConditionalGeneration

# from supervised.int import hf_data
# from supervised.int.gen_subgoal_data import generate_problems
from supervised.int.hf_data import GoalDataset
# from supervised.int.representation.infix_value import DISTANCE_TOKENS
from supervised.rubik import hf_rubik_subgoal, rubik_solver_utils, \
    hf_rubik_value
from supervised.rubik.rubik_solver_utils import cube_to_string, \
    generate_problems_rubik
from utils import hf
from utils import hf_generate
# from visualization.seq_parse import logic_statement_to_seq_string


# class TrivialValueEstimatorRubik:
#     def evaluate(self, state):
#         return -len(cube_to_string(state['observation']['objectives'][0]))
#
#     def construct_networks(self):
#         pass


class ValueEstimatorRubik:
    def __init__(self, checkpoint_path=None, device=None):
        self.checkpoint_path = checkpoint_path
        self.device = device or hf.choose_device()
        self.tokenizer = hf_rubik_value.RubikValueTokenizer(
            model_max_length=hf_rubik_value.SEQUENCE_LENGTH,
            padding_side='right',
            pad_token=hf_rubik_value.PADDING_LEXEME
        )
        self.calls_value = 0

    def construct_networks(self):
        self.model = MBartForConditionalGeneration.from_pretrained(
            self.checkpoint_path
        ).to(self.device)

    def evaluate(self, state):
        return -self.predict_value(state)

    def predict_value(self, state_str):
        self.calls_value += 1
        dataset = GoalDataset.from_policy_input([state_str], self.tokenizer, max_length=56)
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
            decoder_start_token_id=2,  # eos_token_id
            max_length=hf_rubik_subgoal.SEQUENCE_LENGTH,
            num_beams=1,
            num_return_sequences=1,
            num_beam_groups=1,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True
        )

        scores = model_outputs.scores[1].cpu().numpy()[0]
        distance_scores = {}
        expected_distance = 0
        distance_probabilities = {}
        for dist, token_id in hf_rubik_value.DISTANCE_TOKENS.items():
            distance_scores[dist] = scores[token_id]

        softmax_sum = sum([exp(x) for x in distance_scores.values()])
        for dist, score in distance_scores.items():
            dist_p = exp(score)/softmax_sum
            distance_probabilities[dist] = dist_p
            expected_distance += dist * dist_p

        return expected_distance
