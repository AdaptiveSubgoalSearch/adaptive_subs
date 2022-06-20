from supervised.int.utils import theorem_names
from metric_logging import log_text
from supervised.int.representation import base
from supervised.int.representation.infix import VOCABULARY, split_formula_to_lexemes, OBJECTIVE_LEXEME, \
    CONDITION_LEXEME, DESTINATION_LEXEME, BOS_LEXEME, EOS_LEXEME
from visualization import seq_parse
from visualization.seq_parse import entity_to_seq_string

PADDING_LEXEME = '_'
MASK_SEPARATOR_LEXEME = ':'
OUTPUT_START_LEXEME = '@'
INPUT_END_LEXEME = '$'
AXIOM_TOKENS = [char for char in 'ABCDEFGHIJKLMNOPQR']
AXIOM_TO_CHAR = {axiom: char for axiom, char in zip(theorem_names, AXIOM_TOKENS)}
CHAR_TO_AXIOM = {char: axiom for axiom, char in zip(theorem_names, AXIOM_TOKENS)}
AXIOM_LENGTH = {"E": 1, "F": 1, "I": 1, "C": 1, "G": 1, "H": 1, "B": 1, "K": 1, "J": 1, "L": 3, "M": 2, "A": 1, "D": 1}

MASK_TOKENS = [
    '`',
    '~',
    '!',
    ';',
    MASK_SEPARATOR_LEXEME,
]

POLICY_VOCABULARY = (VOCABULARY + AXIOM_TOKENS + MASK_TOKENS)
TOKEN_TO_STR = dict(list(enumerate(POLICY_VOCABULARY)))
STR_TO_TOKEN = {str_: token for token, str_ in TOKEN_TO_STR.items()}
assert len(TOKEN_TO_STR) == len(STR_TO_TOKEN), \
    "There are some duplicated lexemes in vocabulary."
assert STR_TO_TOKEN[BOS_LEXEME] == 0
assert STR_TO_TOKEN[PADDING_LEXEME] == 1
assert STR_TO_TOKEN[EOS_LEXEME] == 2
assert STR_TO_TOKEN[OUTPUT_START_LEXEME] == 3

class EntityMask:
    def __init__(self, entity, left_padding, parsed_name_len, right_padding, ):
        self.entity = entity
        self.left_padding = left_padding
        self.right_padding = right_padding
        self.parsed_name_len = parsed_name_len

    def mask(self):
        return '`'*self.left_padding + \
               '~'*self.parsed_name_len +\
               '`'*self.right_padding

def generate_masks_for_logic_statement(logic_statement):
    first_operand = logic_statement.operands[0]
    second_operand = logic_statement.operands[1]
    first_operand_len = len(split_formula_to_lexemes(entity_to_seq_string(first_operand)))
    second_operand_len = len(split_formula_to_lexemes(entity_to_seq_string(second_operand)))
    separator_len = 1

    first_operand_mask = EntityMask(first_operand, 0, first_operand_len,
                                    separator_len + second_operand_len)
    second_operand_mask = EntityMask(second_operand, first_operand_len + separator_len, second_operand_len,
                                     0)
    operands_with_mask_queue = [first_operand_mask, second_operand_mask]
    entity_to_mask = {
        first_operand: first_operand_mask.mask(),
        second_operand: second_operand_mask.mask()
    }
    mask_to_entity = {
        first_operand_mask.mask(): first_operand,
        second_operand_mask.mask(): second_operand
    }
    while len(operands_with_mask_queue) > 0:
        current_operand = operands_with_mask_queue.pop()
        new_operands = parse_mask_for_entity(current_operand, entity_to_mask, mask_to_entity)
        operands_with_mask_queue.extend(new_operands)

    return entity_to_mask, mask_to_entity


def parse_mask_for_entity(entity_with_mask, entity_to_mask, mask_to_entity):
    entity = entity_with_mask.entity
    left_padding = entity_with_mask.left_padding
    right_padding = entity_with_mask.right_padding
    entity_name = entity_with_mask.entity.name
    operands_with_mask_to_parse = []

    if entity_name.startswith("add") or entity_name.startswith("sub") or entity_name.startswith("mul"):
        first_operand = entity.operands[0]
        second_operand = entity.operands[1]
        first_operand_len = len(split_formula_to_lexemes(entity_to_seq_string(first_operand)))
        second_operand_len = len(split_formula_to_lexemes(entity_to_seq_string(second_operand)))
        first_operand_mask = EntityMask(first_operand, left_padding + 1, first_operand_len, 1+right_padding+second_operand_len+1)
        second_operand_mask = EntityMask(second_operand, left_padding + 1 + first_operand_len + 1, second_operand_len,
                                                 right_padding + 1)

        operands_with_mask_to_parse = [first_operand_mask, second_operand_mask]
        entity_to_mask[first_operand] = first_operand_mask.mask()
        entity_to_mask[second_operand] = second_operand_mask.mask()
        mask_to_entity[first_operand_mask.mask()] = first_operand
        mask_to_entity[second_operand_mask.mask()] = second_operand

    elif entity_name.startswith("opp"):
        operand = entity.operands[0]
        operand_len = len(split_formula_to_lexemes(entity_to_seq_string(operand)))
        operand_mask = EntityMask(operand, left_padding+2, operand_len, right_padding+1)
        operands_with_mask_to_parse = [operand_mask]
        entity_to_mask[operand] = operand_mask.mask()
        mask_to_entity[operand_mask.mask()] = operand

    elif entity_name.startswith("sqr"):
        operand = entity.operands[0]
        operand_len = len(split_formula_to_lexemes(entity_to_seq_string(operand)))
        operand_mask = EntityMask(operand, left_padding + 1, operand_len, right_padding + 2)
        operands_with_mask_to_parse = [operand_mask]
        entity_to_mask[operand] = operand_mask.mask()
        mask_to_entity[operand_mask.mask()] = operand

    elif entity_name.startswith("sqrt"):
        operand = entity.operands[0]
        operand_len = len(split_formula_to_lexemes(entity_to_seq_string(operand)))
        operand_mask = EntityMask(operand, left_padding + 2, operand_len, right_padding + 1)
        operands_with_mask_to_parse = [operand_mask]
        entity_to_mask[operand] = operand_mask.mask()
        mask_to_entity[operand_mask.mask()] = operand

    elif entity_name.startswith("inv"):
        operand = entity.operands[0]
        operand_len = len(split_formula_to_lexemes(entity_to_seq_string(operand)))
        operand_mask = EntityMask(operand, left_padding + 2, operand_len, right_padding + 1)
        operands_with_mask_to_parse = [operand_mask]
        entity_to_mask[operand] = operand_mask.mask()
        mask_to_entity[operand_mask.mask()] = operand
    else:
        entity_len = len(split_formula_to_lexemes(entity_to_seq_string(entity)))
        entity_mask = EntityMask(entity, left_padding, entity_len, right_padding)
        entity_to_mask[entity] = entity_mask.mask()
        mask_to_entity[entity_mask.mask()] = entity

    return operands_with_mask_to_parse

class ActionRepresentationMask(base.Representation):
    #
    # def __init__(self):
    #     self.policy_vocabulary = POLICY_VOCABULARY
    #     self.str_to_token = STR_TO_TOKEN
    token_consts = base.MaskTokenConsts(
        num_tokens=len(STR_TO_TOKEN),
        padding_token=STR_TO_TOKEN[PADDING_LEXEME],
        output_start_token=STR_TO_TOKEN[OUTPUT_START_LEXEME],
        end_token=STR_TO_TOKEN[INPUT_END_LEXEME],
        mask_separator=STR_TO_TOKEN[MASK_SEPARATOR_LEXEME]
    )

    @staticmethod
    def proof_state_to_input_formula(state, add_input_end_lexeme=True):
        conditions = [
            seq_parse.logic_statement_to_seq_string(condition)
            for condition in state['observation']['ground_truth']
        ]
        # most likely only one objective
        objectives = [
            seq_parse.logic_statement_to_seq_string(objective)
            for objective in state['observation']['objectives']
        ]
        formula = OBJECTIVE_LEXEME + OBJECTIVE_LEXEME.join(objectives)
        if len(conditions) > 0:
            formula += CONDITION_LEXEME + CONDITION_LEXEME.join(conditions)
        if add_input_end_lexeme:
            formula += INPUT_END_LEXEME
        return formula

    @classmethod
    def proof_states_to_policy_input_formula(cls, current_state, destination_state):

        formula = cls.proof_state_to_input_formula(current_state, False)
        destination_objectives = [seq_parse.logic_statement_to_seq_string(objective)
            for objective in destination_state['observation']['objectives']
        ]
        formula += DESTINATION_LEXEME + DESTINATION_LEXEME.join(destination_objectives)
        formula += INPUT_END_LEXEME
        return formula

    @classmethod
    def proof_state_to_tokenized_objective(cls, state):
        state_objective = [seq_parse.logic_statement_to_seq_string(objective)
                                  for objective in state['observation']['objectives']
                                  ][0]
        return cls.tokenize_formula(state_objective)

    @classmethod
    def proof_step_and_action_to_formula(cls, proof_step, action):
        '''this version of code assumes there is exactly one objective'''
        objective = proof_step['observation']['objectives'][0]
        return cls.action_to_formula(objective, action)

    @staticmethod
    def proof_step_to_additional_info(proof_step, n_iputs):
        objective = proof_step['observation']['objectives'][0]
        objective_formula = seq_parse.logic_statement_to_seq_string(objective)
        objective_formulas_list = [objective_formula] * n_iputs
        additional_info = OUTPUT_START_LEXEME + MASK_SEPARATOR_LEXEME + \
                          MASK_SEPARATOR_LEXEME.join(objective_formulas_list) + INPUT_END_LEXEME + PADDING_LEXEME
        return additional_info

    @classmethod
    def action_to_formula(cls, objective, action):
        '''this version of code assumes there is exactly one objective'''
        used_axiom = action[0]
        objective_formula = seq_parse.logic_statement_to_seq_string(objective)
        formula = OUTPUT_START_LEXEME + AXIOM_TO_CHAR[used_axiom] + MASK_SEPARATOR_LEXEME
        entity_to_mask, mask_to_entity = generate_masks_for_logic_statement(objective)
        masks = [entity_to_mask[entity] for entity in action[1:]]
        # formula += MASK_SEPARATOR_LEXEME + MASK_SEPARATOR_LEXEME.join(masks) + INPUT_END_LEXEME
        formula += cls.merge_masks(masks) + INPUT_END_LEXEME
        return formula

    @staticmethod
    def merge_masks(masks):
        output_mask = ['`']*len(masks[0])
        for entity_num, entity_mask in enumerate(masks):
            for char_num, char in enumerate(entity_mask):
                if char == '~':
                    assert output_mask[char_num] == '`', 'Masks overlapping'
                    output_mask[char_num] = MASK_TOKENS[entity_num+1]
        return ''.join(output_mask)

    @staticmethod
    def formula_to_action(formula, mask_to_entity):
        params = formula[:-1].split(MASK_SEPARATOR_LEXEME)
        axiom_chosen = CHAR_TO_AXIOM[params[0]]
        input_entities = [mask_to_entity[mask] for mask in params[1:]]
        return (axiom_chosen, *input_entities)

    @staticmethod
    def tokenize_formula(formula):
        lexemes = split_formula_to_lexemes(formula)
        try:
            return [STR_TO_TOKEN[lexeme] for lexeme in lexemes]
        except KeyError as e:
            log_text(
                "Error",
                "Tokenization error - unrecognized lexeme."
                f"Formula split to lexemes:\n{lexemes}"
            )
            raise e


    @staticmethod
    def formula_from_tokens(tokens):
        return "".join(TOKEN_TO_STR[token] for token in tokens)

    @staticmethod
    def action_from_char(char):
        if char in CHAR_TO_AXIOM:
            return CHAR_TO_AXIOM[char]
        else:
            return None
