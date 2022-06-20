from copy import deepcopy

import torch
from transformers import MBartForConditionalGeneration

from supervised.rubik.gen_rubik_data import encode_policy_data, encode_policy_subgoal
from supervised.int.hf_data import GoalDataset
from supervised.rubik import hf_rubik_value, hf_rubik_policy, gen_rubik_data, \
    rubik_solver_utils
from supervised.rubik.rubik_solver_utils import make_RubikEnv, cube_to_string, \
    generate_problems_rubik
from utils import hf
from utils import hf_generate
import numpy as np


class VerificatorRubik:
    def __init__(self,
                 checkpoint_path=None,
                 device=None):
        self.checkpoint_path = checkpoint_path
        self.device = device or hf.choose_device()
        self.act_rep = None
        self.tokenizer = hf_rubik_policy.RubikPolicyTokenizer(
            model_max_length=hf_rubik_policy.SEQUENCE_LENGTH,
            padding_side='right',
            pad_token=hf_rubik_policy.PADDING_LEXEME
        )
        self.model = None

    def construct_networks(self):
        self.model = MBartForConditionalGeneration.from_pretrained(
            self.checkpoint_path
        ).to(self.device)

    def predict_reachability(self, state, subgoal):
        verificator_input = encode_policy_subgoal(state, subgoal)
        dataset = GoalDataset.from_state([verificator_input], self.tokenizer,
                                         max_length=hf_rubik_policy.SEQUENCE_LENGTH)

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
            max_length=hf_rubik_policy.SEQUENCE_LENGTH,
            num_beams=1,
            num_return_sequences=1,
            num_beam_groups=1,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True
        )

        scores = model_outputs.scores[1].cpu().numpy()[0]
        reachability_probs = np.array([0., 0.])
        for is_reached, token_id in hf_rubik_policy.VERIFICATOR_OUTPUT_TOKENS.items():
            token_id = hf_rubik_policy.STR_TO_TOKEN[token_id]
            reachability_probs[is_reached] = np.exp(scores[token_id])
        print('reachability raw probs:', reachability_probs)

        if sum(reachability_probs) < 0.01:
            print('WARNING: inconsistent answer')
            prob = 0.5  # totally uncertain
        else:
            prob = reachability_probs[1] / sum(reachability_probs)

        print('predicted probability:', prob)
        return prob