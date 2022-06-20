from copy import deepcopy

import torch
import transformers
from transformers import MBartForConditionalGeneration

# from envs.int.theorem_prover_env import TheoremProverEnv
# from policies.policy_utils import decode_prediction
# from supervised import ActionRepresentationPointer
# from supervised.int import ActionRepresentationMask
# from supervised.int.gen_subgoal_data import generate_problems
# from supervised.int.hf_data import GoalDataset, IntPolicyTokenizer, IntGoalTokenizer, IntPolicyTokenizerMask, \
#     IntPolicyTokenizerPointer
# from supervised.int.representation import infix
# from supervised.int.representation.action_representation_pointer import split_formula_to_lexemes, CHAR_TO_AXIOM, \
#     AXIOM_LENGTH, POINTER_SYMBOLS, generate_masks_for_logic_statement
from supervised.int.hf_data import GoalDataset
from supervised.rubik import hf_rubik_policy, rubik_solver_utils, gen_rubik_data
from supervised.rubik.gen_rubik_data import encode_policy_data
from supervised.rubik.rubik_solver_utils import make_RubikEnv, cube_to_string, \
    generate_problems_rubik
from utils import hf_generate, hf
# from visualization.seq_parse import entity_to_seq_string, logic_statement_to_seq_string


class VanillaPolicyRubik:
    def __init__(self, checkpoint_path, device=None, n_actions=None, num_beams=None, temperature=None):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.device = device or hf.choose_device()
        self.tokenizer = hf_rubik_policy.RubikPolicyTokenizer(
            model_max_length=hf_rubik_policy.SEQUENCE_LENGTH,
            padding_side='right',
            pad_token=hf_rubik_policy.PADDING_LEXEME
        )
        self.env = make_RubikEnv()
        self.n_actions = n_actions
        self.num_beams = num_beams
        self.temperature = temperature

    def construct_networks(self):
        self.model = MBartForConditionalGeneration.from_pretrained(
            self.checkpoint_path
        ).to(self.device)

    def build_goals(self, state):
        actions, debugg_info = self.predict_actions(state, self.num_beams, self.n_actions)
        goals = []

        for action in actions:
            action = action[0]

            self.env.load_state(gen_rubik_data.cube_str_to_state(state[1:-1]))
            new_obs, _, done, _ = self.env.step(action)
            new_obs_str = cube_to_string(new_obs)

            if done:
                return goals, (new_obs_str, [action], done)

            goals.append((new_obs_str, [action], done))

        return goals, None


    def predict_action_str(self, state_formula, num_beams, num_return_sequences):
        policy_input = encode_policy_data(state_formula[1:-1],
                                          state_formula[1:-1], 0)[0]
        dataset = GoalDataset.from_policy_input([policy_input], self.tokenizer, max_length=56)
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
            attention_mask=torch.tensor(
                [input.attention_mask for input in inputs],
                dtype=torch.int64,
                device=self.model.device,
            ),
            decoder_start_token_id=2,  # eos_token_id
            num_beams=num_beams,
            # num_beam_groups=2,
            num_return_sequences=num_return_sequences,
            max_length=hf_rubik_policy.SEQUENCE_LENGTH,
            temperature=self.temperature
        )

        return [
            self.tokenizer.decode(output.cpu())
            for output in model_outputs
        ]

    def predict_actions(self, proof_state, num_beams, num_return_sequences):
        debug_info = {}
        actions_str = self.predict_action_str(proof_state, num_beams, num_return_sequences)

        # _, mask_to_entity = generate_masks_for_logic_statement(proof_state['observation']['objectives'][0])
        # actions_str = self.predict_action_str(state_formula, num_beams, num_return_sequences)

        # print(f'state formula = {proof_state} | action_str = {actions_str}')
        full_actions = []
        for action_str in actions_str:
            decoded_action = rubik_solver_utils.decode_action(action_str)

            if decoded_action is not None:
                proof_state_copy = deepcopy(proof_state)
                full_actions.append((decoded_action, proof_state_copy))

        # print(f'debugg = {debug_info}')
        # print(f'full actions = {[x[0] for x in full_actions]}')
        # if len(full_actions)>1:
        #     full_actions = [full_actions[0]]
        return full_actions, debug_info


    def solve(self, proof_state, debugg_mode=False):
        path = []
        reaching_goal_debug_info = {}
        self.env.load_state(gen_rubik_data.cube_str_to_state(proof_state[1:-1]))
        subgoal_reached = False
        steps_taken = 0
        current_proof_state = deepcopy(proof_state)
        seen_states = {proof_state}
        while not subgoal_reached and steps_taken < 100:
            if debugg_mode:
                print(f'Step = {steps_taken} \n curr = {current_proof_state}')
            actions, debugg_info = self.predict_actions(current_proof_state, 32, 1)
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
                        print(f'{action} | ()')
                    new_obs, _, done, _ = self.env.step(action)

                    new_obs_str = cube_to_string(new_obs)
                    new_obs = new_obs_str

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
        return False, {'limit': True}
