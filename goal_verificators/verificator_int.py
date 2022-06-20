from copy import deepcopy

import torch
from transformers import MBartForConditionalGeneration

from supervised.int import hf_data
from supervised.int.representation import action_representation_pointer
from supervised.int.hf_data import GoalDataset
from utils import hf
from utils import hf_generate
import numpy as np
from supervised.int.representation.action_representation_mask import PADDING_LEXEME


class VerificatorInt:
    def __init__(self,
                 checkpoint_path=None,
                 device=None):
        self.checkpoint_path = checkpoint_path
        self.device = device or hf.choose_device()
        self.act_rep = None
        self.tokenizer = hf_data.IntPolicyTokenizerPointer(
            model_max_length=512,
            padding_side='right',
            pad_token=PADDING_LEXEME,
        )
        self.model = None

    def construct_networks(self):
        self.model = MBartForConditionalGeneration.from_pretrained(
            self.checkpoint_path
        ).to(self.device)

    def predict_reachability(self, state_subgoal):
        dataset = GoalDataset.from_state([state_subgoal], self.tokenizer, max_length=512)

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
            max_length=512,
            num_beams=1,
            num_return_sequences=1,
            num_beam_groups=1,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True
        )

        scores = model_outputs.scores[1].cpu().numpy()[0]
        reachability_probs = np.array([0., 0.])
        for is_reached, token in enumerate(action_representation_pointer.VERIFICATOR_TOKENS):
            token_id = action_representation_pointer.STR_TO_TOKEN[token]
            reachability_probs[is_reached] = np.exp(scores[token_id])
        print('reachability raw probs:', reachability_probs)

        if sum(reachability_probs) < 0.01:
            print('WARNING: totally uncertain')
            prob = 0.5  # totally uncertain
        else:
            prob = reachability_probs[1] / sum(reachability_probs)

        print('predicted probability:', prob)
        return prob
