import string

from supervised.int.representation import base

PADDING_WORD = '-'
OUTPUT_START_WORD = 'OUTPUT:'

VOCABULARY = (
        [
            PADDING_WORD,
            '#',
            '&',
            '$',
            OUTPUT_START_WORD,
            'Equivalent',
            '(',
            ')',
            ',',
            'mul',
            'add',
            'inv',
            '0',
            '1',
            'opp',
            'sub',
            'sqr',
            'sqrt',
            'BiggerOrEqual',
            'SmallerOrEqual',
            'NonNegative',
        ] +
        list(string.ascii_lowercase)
)

TOKEN_TO_STR = dict(list(enumerate(VOCABULARY)))
STR_TO_TOKEN = {str_: token for token, str_ in TOKEN_TO_STR.items()}
assert len(TOKEN_TO_STR) == len(STR_TO_TOKEN), \
    "There are some duplicated lexemes in vocabulary."


class PrefixRepresentation(base.Representation):
    token_consts = base.TokenConsts(
        num_tokens=len(STR_TO_TOKEN),
        padding_token=STR_TO_TOKEN[PADDING_WORD],
        output_start_token=STR_TO_TOKEN[OUTPUT_START_WORD],
    )

    @staticmethod
    def proof_state_to_input_formula(state):
        conditions = [
            condition.to_string()
            for condition in state['observation']['ground_truth']
        ]
        # most likely only one objective
        objectives = [
            objective.to_string()
            for objective in state['observation']['objectives']
        ]
        formula = '# ' + ' # '.join(objectives) + ' '
        if len(conditions) > 0:
            formula += '& ' + ' & '.join(conditions) + ' '
        formula += '$'
        return formula

    @staticmethod
    def proof_state_to_target_formula(state):
        objectives = [
            objective.to_string()
            for objective in state['observation']['objectives']
        ]
        return OUTPUT_START_WORD + ' ' + ' # '.join(objectives)

    @staticmethod
    def tokenize_formula(formula):
        return [STR_TO_TOKEN[str_] for str_ in formula.split(" ")]

    @staticmethod
    def formula_from_tokens(tokens):
        return " ".join(TOKEN_TO_STR[token] for token in tokens)
