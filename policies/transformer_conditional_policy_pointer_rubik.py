from copy import deepcopy

import torch
from transformers import MBartForConditionalGeneration

# from envs.int.theorem_prover_env import TheoremProverEnv
# from policies.policy_utils import decode_prediction
# from supervised import ActionRepresentationPointer
# from supervised.int.gen_subgoal_data import generate_problems
# from supervised.int.hf_data import GoalDataset, IntPolicyTokenizerPointer
# from supervised.int.representation import infix
# from supervised.int.representation.action_representation_pointer import generate_masks_for_logic_statement
from supervised.rubik.gen_rubik_data import encode_policy_data
from third_party.INT import visualization
from supervised.int import hf_data
from supervised.int.hf_data import GoalDataset
from supervised.rubik import hf_rubik_value, hf_rubik_policy, gen_rubik_data, \
    rubik_solver_utils
from supervised.rubik.rubik_solver_utils import make_RubikEnv, cube_to_string, \
    generate_problems_rubik
from utils import hf
from utils import hf_generate
# from visualization.seq_parse import entity_to_seq_string, logic_statement_to_seq_string


class ConditionalPolicyRubik:
    def __init__(self,
                 checkpoint_path=None,
                 max_steps=None,
                 device=None):
        self.checkpoint_path = checkpoint_path
        self.max_steps = max_steps
        self.device = device or hf.choose_device()
        self.act_rep = None #ActionRepresentationPointer()
        self.tokenizer = hf_rubik_policy.RubikPolicyTokenizer(
            model_max_length=hf_rubik_policy.SEQUENCE_LENGTH,
            padding_side='right',
            pad_token=hf_rubik_policy.PADDING_LEXEME
        )
        self.env = make_RubikEnv()
        self.model = None

    def construct_networks(self):
        self.model = MBartForConditionalGeneration.from_pretrained(
            self.checkpoint_path
        ).to(self.device)

    def predict_action_str(self, state_target_formula):
        policy_input = encode_policy_data(state_target_formula[0][1:-1], state_target_formula[1][1:-1], 0)[0]
        dataset = GoalDataset.from_state([policy_input], self.tokenizer, max_length=hf_rubik_policy.SEQUENCE_LENGTH)
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
            max_length=hf_rubik_value.SEQUENCE_LENGTH,
            # num_beams=1,
            # # num_beam_groups=2,
            # num_return_sequences=1,
            # temperature=0.001
        )
        return [
            self.tokenizer.decode(output.cpu())
            for output in model_outputs
        ]

    def predict_actions(self, proof_state, subgoal):
        debug_info = {}
        state_target_formula = (proof_state, subgoal)
        try:
            action_str = self.predict_action_str(state_target_formula)[0]
        except KeyError:
            return None
        return rubik_solver_utils.decode_action(action_str)

    def is_solved(self, proof_state):
        return proof_state == cube_to_string(self.env.solved_obs)

    def reach_subgoal(self, proof_state, subgoal, debugg_mode=False, steps_counter=[0], force_step_limit=None):
        # print(f'reaching subgoal: {proof_state} -> {subgoal}')
        path = []
        reaching_goal_debug_info = {}
        self.env.load_state(gen_rubik_data.cube_str_to_state(proof_state[1:-1]))
        subgoal_reached = False
        steps_taken = 0
        current_proof_state = deepcopy(proof_state)
        seen_states = {current_proof_state}

        if force_step_limit is None:
            step_limit = self.max_steps
        else:
            step_limit = force_step_limit

        while not subgoal_reached and steps_taken < step_limit:
            action = self.predict_actions(current_proof_state, subgoal)
            steps_counter[0] += 1
            path.append(action)

            if action is None:
                reaching_goal_debug_info['None action'] = True
                if debugg_mode:
                    print('None action')
                return False, [], False, subgoal
            else:
                if debugg_mode:
                    # print(f'{action[0]} | {[entity_to_seq_string(x) for x in action[1:]]}')
                    print(f'{action} | ()')
                new_obs, _, done, _ = self.env.step(action)
                current_proof_state = deepcopy(new_obs)
                current_proof_state_str = cube_to_string(current_proof_state)
                current_proof_state = current_proof_state_str
                if current_proof_state_str in seen_states:
                    return False, [], False, subgoal
                else:
                    seen_states.add(current_proof_state_str)
                # current_proof_state = {'observation': deepcopy(new_obs)}
                if isinstance(subgoal, str):
                    subgoal_str = subgoal
                else:
                    subgoal_str = cube_to_string(subgoal)
                if current_proof_state == subgoal_str:
                    return True, path, done, current_proof_state
            steps_taken += 1

        return False, path, False, subgoal
