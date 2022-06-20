from copy import deepcopy

import torch
from transformers import MBartForConditionalGeneration

from envs.int.theorem_prover_env import TheoremProverEnv
from policies.policy_utils import decode_prediction
from supervised import ActionRepresentationPointer
from supervised.int.hf_data import GoalDataset, IntPolicyTokenizerPointer
from supervised.int.representation import infix
from supervised.int.representation.action_representation_pointer import generate_masks_for_logic_statement
from utils import hf_generate, hf
from visualization.seq_parse import entity_to_seq_string, logic_statement_to_seq_string


class VanillaPolicyINT:
    def __init__(self, checkpoint_path, num_return_sequences, max_steps_allowed, num_beams, device=None):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.device = device or hf.choose_device()
        self.act_rep = ActionRepresentationPointer()
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.max_steps_allowed = max_steps_allowed
        self.tokenizer = IntPolicyTokenizerPointer(
            model_max_length=512,
            padding_side='right',
            pad_token=infix.PADDING_LEXEME
        )
        self.env = TheoremProverEnv()
        self.prediction_counter = 0

    def reset_counter(self):
        self.prediction_counter = 0

    def read_counter(self):
        return self.prediction_counter

    def construct_networks(self):
        self.model = MBartForConditionalGeneration.from_pretrained(
            self.checkpoint_path
        ).to(self.device)

    def predict_action_str(self, state_formula):
        dataset = GoalDataset.from_policy_input([state_formula], self.tokenizer, max_length=512)
        inputs = [
            hf_generate.GenerationInput(
                input_ids=entry['input_ids'],
                attention_mask=entry['attention_mask']
            )
            for entry in dataset
        ]

        length_penalty = 1.0  # Default and recommended value for beam_search
        [sequences], [scores] = hf_generate.generate_sequences(
            model=self.model,
            inputs=inputs,
            num_return_sequences=self.num_return_sequences,
            num_beams=self.num_beams,
            max_length=512,
            length_penalty=length_penalty,
        )
        sequences_strs = [
            self.tokenizer.decode(sequence).replace('_', '')
            for sequence in sequences
        ]
        sequences_lengths = list(map(len, sequences_strs))
        probs = hf_generate.compute_probabilities(
            scores, sequences_lengths, length_penalty=length_penalty
        )
        return list(zip(sequences_strs, probs.tolist()))

    def predict_actions(self, proof_state):
        self.prediction_counter += 1
        debug_info = {}
        state_formula = self.act_rep.proof_states_to_policy_input_formula(
            current_state=proof_state,
            destination_state='',
            vanilla=True
        )
        actions_probs = self.predict_action_str(state_formula)

        print(f'state formula = {state_formula} | actions_probs = {actions_probs}')
        full_actions = []
        for action_str, prob in actions_probs:
            decoded_action = decode_prediction(action_str)
            if decoded_action is not None:
                proof_state_copy = deepcopy(proof_state)
                _, mask_to_entity = generate_masks_for_logic_statement(proof_state_copy['observation']['objectives'][0])
                success = True
                full_action = [decoded_action[0]]
                debug_info['proper masks'] = True
                for entity_str in decoded_action[1:]:
                    if entity_str in mask_to_entity:
                        full_action.append(mask_to_entity[entity_str])
                    else:
                        debug_info['proper masks'] = False
                        success = False
                        break
                if success:
                    full_actions.append((full_action, proof_state_copy, prob))

        return full_actions, debug_info

    def solve(self, proof_state, debugg_mode=False):
        path = []
        reaching_goal_debug_info = {}
        self.env.load_problem_step(proof_state)
        subgoal_reached = False
        steps_taken = 0
        current_proof_state = deepcopy(proof_state)
        seen_states = {logic_statement_to_seq_string(proof_state['observation']['objectives'][0])}
        while not subgoal_reached and len(seen_states) < self.max_steps_allowed:
            if debugg_mode:
                print(
                    f'Step = {steps_taken} \n curr = {logic_statement_to_seq_string(current_proof_state["observation"]["objectives"][0])}')
            actions, debugg_info = self.predict_actions(current_proof_state)
            debugg_info['num_steps'] = len(seen_states)
            if len(actions) > 0:
                action = actions[0][0]
                path.append(action)
                if action is None:
                    reaching_goal_debug_info['None action'] = True
                    if debugg_mode:
                        print('None action')
                    return False
                else:
                    if debugg_mode:
                        print(f'{action[0]} | {[entity_to_seq_string(x) for x in action[1:]]}')
                    new_obs, _, done, _ = self.env.step(action)
                    if len(new_obs['observation']['objectives']) > 1:
                        debugg_info['more_obj'] = True
                        print('more obj')
                        return False, debugg_info

                    new_obs_str = logic_statement_to_seq_string(new_obs['observation']['objectives'][-1])
                    if new_obs_str in seen_states and not done:
                        print('Seen state')
                        debugg_info['seen_state'] = True
                        return False, debugg_info

                    seen_states.add(new_obs_str)
                    current_proof_state = deepcopy(new_obs)
                    if done:
                        return True, debugg_info
                steps_taken += 1
            else:
                steps_taken += 1
                break
        return False, {'limit': True, 'num_steps': len(seen_states)}
