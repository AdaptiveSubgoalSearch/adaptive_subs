from visualization.seq_parse import entity_to_seq_string


class EntityMask:
    def __init__(self, entity, left_padding, parsed_name_len, right_padding, ):
        self.entity = entity
        self.left_padding = left_padding
        self.right_padding = right_padding
        self.parsed_name_len = parsed_name_len

    def mask(self):
        return '0'*self.left_padding + '1'*self.parsed_name_len+ '0'*self.right_padding

def generate_masks_for_logic_statement(logic_statement):
    logic_statement_name = logic_statement.name
    first_operand = logic_statement.operands[0]
    second_operand = logic_statement.operands[1]
    first_operand_len = len(entity_to_seq_string(first_operand))
    second_operand_len = len(entity_to_seq_string(second_operand))
    separator_len = 0
    if logic_statement_name.startswith("BiggerOrEqual"):
        separator_len = 4
    elif logic_statement_name.startswith("SmallerOrEqual"):
        separator_len = 4
    elif logic_statement_name.startswith("Equivalent"):
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
        first_operand_len = len(entity_to_seq_string(first_operand))
        second_operand_len = len(entity_to_seq_string(second_operand))
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
        operand_len = len(entity_to_seq_string(operand))
        operand_mask = EntityMask(operand, left_padding+2, operand_len, right_padding+1)
        operands_with_mask_to_parse = [operand_mask]
        entity_to_mask[operand] = operand_mask.mask()
        mask_to_entity[operand_mask.mask()] = operand

    elif entity_name.startswith("sqr"):
        operand = entity.operands[0]
        operand_len = len(entity_to_seq_string(operand))
        operand_mask = EntityMask(operand, left_padding + 1, operand_len, right_padding + 3)
        operands_with_mask_to_parse = [operand_mask]
        entity_to_mask[operand] = operand_mask.mask()
        mask_to_entity[operand_mask.mask()] = operand

    elif entity_name.startswith("sqrt"):
        operand = entity.operands[0]
        operand_len = len(entity_to_seq_string(operand))
        operand_mask = EntityMask(operand, left_padding + 6, operand_len, right_padding + 2)
        operands_with_mask_to_parse = [operand_mask]
        entity_to_mask[operand] = operand_mask.mask()
        mask_to_entity[operand_mask.mask()] = operand

    elif entity_name.startswith("inv"):
        operand = entity.operands[0]
        operand_len = len(entity_to_seq_string(operand))
        operand_mask = EntityMask(operand, left_padding + 3, operand_len, right_padding + 1)
        operands_with_mask_to_parse = [operand_mask]
        entity_to_mask[operand] = operand_mask.mask()
        mask_to_entity[operand_mask.mask()] = operand
    else:
        entity_len = len(entity_to_seq_string(entity))
        entity_mask = EntityMask(entity, left_padding, entity_len, right_padding)
        entity_to_mask[entity] = entity_mask.mask()
        mask_to_entity[entity_mask.mask()] = entity

    return operands_with_mask_to_parse