from supervised.int.representation.action_representation_pointer import CHAR_TO_AXIOM, AXIOM_LENGTH, POINTER_SYMBOLS


def decode_prediction(prediction):
    if prediction[0:2] != '$@' or prediction[-1:] != '$':
        # Invalid prediction format
        return None
    prediction_str = prediction[2:-1]
    if len(prediction_str) == 0:
        return None

    if prediction_str[0] in CHAR_TO_AXIOM:
        axiom = CHAR_TO_AXIOM[prediction_str[0]]
        axiom_len = AXIOM_LENGTH[prediction_str[0]]
        input_entities_raw = [prediction_str[1:] for _ in range(axiom_len)]
        input_entities_str = []
        for num in range(len(input_entities_raw)):
            entity_str = input_entities_raw[num]
            pointer_symbol = POINTER_SYMBOLS[num]
            for different_pointer_symbol in POINTER_SYMBOLS:
                if different_pointer_symbol != pointer_symbol:
                    entity_str = entity_str.replace(different_pointer_symbol, '')
            entity_str = entity_str.replace(pointer_symbol, POINTER_SYMBOLS[0])
            input_entities_str.append(entity_str)
        return [axiom, *input_entities_str]
    else:
        return None