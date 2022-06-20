from supervised.rubik import gen_rubik_data

import transformers
from typing import Dict, List, Optional, Tuple

from supervised.rubik.gen_rubik_data import policy_encoding

PADDING_LEXEME = '_'
EOS_LEXEME = '$'
OUTPUT_START_LEXEME = '@'
BOS_LEXEME = '?'

VOCABULARY = [
    BOS_LEXEME,
    PADDING_LEXEME,
    EOS_LEXEME,
    OUTPUT_START_LEXEME,
] + [chr(i) for i in range(65, 65+25)] + [chr(i) for i in range(97, 97+25)]

TOKEN_TO_STR = dict(list(enumerate(VOCABULARY)))
STR_TO_TOKEN = {str_: token for token, str_ in TOKEN_TO_STR.items()}
assert len(TOKEN_TO_STR) == len(STR_TO_TOKEN), \
    "There are some duplicated lexemes in vocabulary."

assert STR_TO_TOKEN[BOS_LEXEME] == 0
assert STR_TO_TOKEN[PADDING_LEXEME] == 1
assert STR_TO_TOKEN[EOS_LEXEME] == 2
assert STR_TO_TOKEN[OUTPUT_START_LEXEME] == 3

assert len(TOKEN_TO_STR) == 4 + 25 + 25

SEQUENCE_LENGTH = 56

VERIFICATOR_OUTPUT_TOKENS = {i: policy_encoding()[1][i] for i in range(2)}


class RubikPolicyTokenizer(transformers.PreTrainedTokenizer):
    model_input_names = ['input_ids', 'attention_mask']

    @property
    def vocab_size(self) -> int:
        return len(STR_TO_TOKEN)

    def _tokenize(self, text, **kwargs):
        return [c for c in text]

    def _convert_token_to_id(self, token):
        return STR_TO_TOKEN[token]

    def _convert_id_to_token(self, index: int) -> str:
        return TOKEN_TO_STR[index]

    def get_vocab(self) -> Dict[str, int]:
        return STR_TO_TOKEN

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        # We don't train the tokenizer, so there is no point in saving vocabulary.
        return ('/dev/null',)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return ''.join(tokens)

    def check_validity(self, data):
        return gen_rubik_data.check_valid(data)
