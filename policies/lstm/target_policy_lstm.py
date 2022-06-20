from logic.logic import LogicStatement
from policies.lstm.lstm_policy_architecture import get_encoder_decoder_models, get_lstm_goal_predictor, compile_lstm, \
    state_destination_formula_to_encoder_input, sample_output_sequence
from supervised.int import ActionRepresentationMask
from supervised.int.representation.action_representation_mask import generate_masks_for_logic_statement
from visualization.seq_parse import logic_statement_to_seq_string


class ConditionalLowLevelPolicyLSTM:

    def __init__(self):
        self.representation =  ActionRepresentation
        self.token_consts = self.representation.token_consts
        self.encoder = None
        self.decoder = None

    def load_model(self, model_id):
        self.encoder, self.decoder = get_encoder_decoder_models(
            self.token_consts, checkpoint_path=model_id, model=None
        )

    def construct_model(self, latent_dim=256, learning_rate=0.0001):
        self.full_model = get_lstm_goal_predictor(latent_dim, self.token_consts)
        compile_lstm(self.full_model, self.token_consts, learning_rate)
        self.encoder, self.decoder = get_encoder_decoder_models(
            self.token_consts, checkpoint_path=None, model=self.full_model
        )

    def act(self, observation, target):
        """

        :param observation: dictionary of form {'objectives': }
        :param target: target state (in form of string or logic_statement)
        :return: tuple describing the chosen action
        """
        if isinstance(target, LogicStatement):
            target = logic_statement_to_seq_string(target)
        elif isinstance(target, str):
            pass
        else:
            raise ValueError(f'Unsupported target type, expected string or LogicStatement got {type(target)}.')


    def predict_action_formula(self, state_destination_formula):
        assert self.encoder is not None and self. decoder is not None, "You must load model before predicting."
        encoder_input = state_destination_formula_to_encoder_input(
            state_destination_formula, self.representation
        )
        predicted_tokenized_formula = sample_output_sequence(
            encoder_input,
            encoder_model=self.encoder,
            decoder_model=self.decoder,
            token_consts=self.token_consts
        )
        return self.representation.formula_from_tokens(
                predicted_tokenized_formula
            )

    def state_destination_to_action(self, current_state, destination_state):
        model_input = self.representation.proof_states_to_policy_input_formula(current_state, destination_state)
        entity_to_mask, mask_to_entity = generate_masks_for_logic_statement(current_state['observation']['objectives'][0])
        print(mask_to_entity)
        action_formula = self.predict_action_formula(model_input)
        action_formula = action_formula.replace('$', '')
        action_formula = action_formula.replace('_', '')
        action_elements = action_formula.split(':')
        print(model_input)
        print(action_elements)
        action = None
        axiom_chosen = self.representation.action_from_char(action_elements[0])
        input_entities = []
        success = True
        if axiom_chosen is not None:
            for mask in action_elements[1:]:
                if mask in mask_to_entity:
                    input_entities.append(mask_to_entity[mask])
                else:
                    success = False
        if axiom_chosen is not None:
            return (axiom_chosen, *input_entities), success


