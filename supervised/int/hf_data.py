"""Huggingface classes for using INT data."""

import math

from typing import Dict, List, Optional, Tuple

from torch.utils import data as torch_data
import transformers

from supervised.int import gen_subgoal_data
from supervised.int.representation import infix as int_repr
from supervised.int.representation import action_representation_pointer as act_rep
from supervised.int.representation import action_representation_mask as act_rep_mask
from supervised.int.representation import action_representation_pointer as act_rep_pointer
from supervised.int.representation import infix_value as infix_value_rep
from utils import hf as hf_utils


class GoalDataset(torch_data.Dataset):
    def __init__(self, data):
        self._data = data
        self._size = len(self._data['input_ids'])

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return {
            key: value[idx]
            for key, value in self._data.data.items()
        }

    @staticmethod
    def from_formula_pairs(pairs, tokenizer, max_length=512):
        inputs, targets = zip(*pairs)
        model_inputs = tokenizer(
            inputs, max_length=max_length,
            padding=True, truncation=True
        )

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_length,
                padding=True, truncation=True
            )
        model_inputs["labels"] = labels["input_ids"]

        return GoalDataset(model_inputs)

    @staticmethod
    def from_state(formulas, tokenizer, max_length):
        model_inputs = tokenizer(
            formulas, max_length=max_length,
            padding=False, truncation=True
        )
        return GoalDataset(model_inputs)

    @staticmethod
    def from_policy_input(formulas, tokenizer, max_length, padding=False):
        model_inputs = tokenizer(
            formulas, max_length=max_length,
            padding=padding, truncation=True
        )
        return GoalDataset(model_inputs)


class Subset(torch_data.Dataset):
    def __init__(self, dataset, ratio):
        self._dataset = dataset
        self._size = math.ceil(len(dataset) * ratio)
        assert 0 <= self._size <= len(dataset)

    def __len__(self):
        return self._size

    def __getitem__(self, item):
        if item >= self._size:
            raise IndexError()
        return self._dataset[item]


class IntGoalTokenizer(transformers.PreTrainedTokenizer):
    model_input_names = ['input_ids', 'attention_mask']

    @property
    def vocab_size(self) -> int:
        return len(int_repr.STR_TO_TOKEN)

    def _tokenize(self, text, **kwargs):
        return int_repr.split_formula_to_lexemes(text)

    def _convert_token_to_id(self, token):
        return int_repr.STR_TO_TOKEN[token]

    def _convert_id_to_token(self, index: int) -> str:
        return int_repr.TOKEN_TO_STR[index]

    def get_vocab(self) -> Dict[str, int]:
        return int_repr.STR_TO_TOKEN

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        # We don't train the tokenizer, so there is no point in saving vocabulary.
        return ('/dev/null',)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return ''.join(tokens)


class IntPolicyTokenizer(transformers.PreTrainedTokenizer):
    model_input_names = ['input_ids', 'attention_mask']

    @property
    def vocab_size(self) -> int:
        return len(act_rep.STR_TO_TOKEN)

    def _tokenize(self, text, **kwargs):
        return act_rep.split_formula_to_lexemes(text)

    def _convert_token_to_id(self, token):
        return act_rep.STR_TO_TOKEN[token]

    def _convert_id_to_token(self, index: int) -> str:
        return act_rep.TOKEN_TO_STR[index]

    def get_vocab(self) -> Dict[str, int]:
        return act_rep.STR_TO_TOKEN

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        # We don't train the tokenizer, so there is no point in saving vocabulary.
        return ('/dev/null',)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return ''.join(tokens)

class IntPolicyTokenizerMask(transformers.PreTrainedTokenizer):
    model_input_names = ['input_ids', 'attention_mask']

    @property
    def vocab_size(self) -> int:
        return len(act_rep_mask.STR_TO_TOKEN)

    def _tokenize(self, text, **kwargs):
        return act_rep_mask.split_formula_to_lexemes(text)

    def _convert_token_to_id(self, token):
        return act_rep_mask.STR_TO_TOKEN[token]

    def _convert_id_to_token(self, index: int) -> str:
        return act_rep_mask.TOKEN_TO_STR[index]

    def get_vocab(self) -> Dict[str, int]:
        return act_rep_mask.STR_TO_TOKEN

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        # We don't train the tokenizer, so there is no point in saving vocabulary.
        return ('/dev/null',)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return ''.join(tokens)

class IntValueTokenizer(transformers.PreTrainedTokenizer):
    model_input_names = ['input_ids', 'attention_mask']

    @property
    def vocab_size(self) -> int:
        return len(infix_value_rep.STR_TO_TOKEN)

    def _tokenize(self, text, **kwargs):
        return infix_value_rep.split_formula_to_lexemes(text)

    def _convert_token_to_id(self, token):
        return infix_value_rep.STR_TO_TOKEN[token]

    def _convert_id_to_token(self, index: int) -> str:
        return infix_value_rep.TOKEN_TO_STR[index]

    def get_vocab(self) -> Dict[str, int]:
        return infix_value_rep.STR_TO_TOKEN

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        # We don't train the tokenizer, so there is no point in saving vocabulary.
        return ('/dev/null',)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return ''.join(tokens)

class IntPolicyTokenizerPointer(transformers.PreTrainedTokenizer):
    model_input_names = ['input_ids', 'attention_mask']

    @property
    def vocab_size(self) -> int:
        return len(act_rep_pointer.STR_TO_TOKEN)

    def _tokenize(self, text, **kwargs):
        return act_rep_pointer.split_formula_to_lexemes(text)

    def _convert_token_to_id(self, token):
        assert token in act_rep_pointer.STR_TO_TOKEN, (
            f"Token '{token}' is out of vocabulary."
        )
        return act_rep_pointer.STR_TO_TOKEN[token]

    def _convert_id_to_token(self, index: int) -> str:
        return act_rep_pointer.TOKEN_TO_STR[index]

    def get_vocab(self) -> Dict[str, int]:
        return act_rep_pointer.STR_TO_TOKEN

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        # We don't train the tokenizer, so there is no point in saving vocabulary.
        return ('/dev/null',)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return ''.join(tokens)

def generate_int_goal_dataset(
        n_proofs, tokenizer, max_seq_length,
        representation, kl_dict,
        done_epochs, log_prefix
):
    formula_pairs = gen_subgoal_data.generate_formula_pairs(
        n_proofs, representation
    )
    hf_utils.log_formula_statistics(
        formula_pairs, done_epochs, log_prefix,
        threshold=max_seq_length
    )
    return GoalDataset.from_formula_pairs(
        formula_pairs, tokenizer, max_length=max_seq_length
    )
