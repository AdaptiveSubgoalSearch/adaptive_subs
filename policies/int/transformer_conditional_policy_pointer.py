import time
from copy import deepcopy
import collections

import gin
from transformers import MBartForConditionalGeneration

from envs.int.theorem_prover_env import TheoremProverEnv
from policies.policy_utils import decode_prediction
from supervised import ActionRepresentationPointer
from supervised.int.hf_data import GoalDataset, IntPolicyTokenizerPointer
from supervised.int.representation import infix
from supervised.int.representation.action_representation_pointer import generate_masks_for_logic_statement
from supervised.int import utils as int_utils
from utils import hf
from utils import hf_generate
from visualization.seq_parse import logic_statement_to_seq_string


class SubgoalPursuitData:
    def __init__(
        self, current_state, actions, intermediate_states, subgoal_str,
        subgoal_reached, seen_states, is_finished, done
    ):
        self.current_state = current_state
        self.actions = actions
        self.intermediate_states = intermediate_states
        self.subgoal_str = subgoal_str
        self.subgoal_reached = subgoal_reached
        self.seen_states = seen_states
        self.is_finished = is_finished
        self.done = done


SubgoalPath = collections.namedtuple('SubgoalPath', [
    'actions',
    'intermediate_states',
    'subgoal_state',
    'done',
])


class ConditionalPolicyINT:
    def __init__(self,
                 checkpoint_path=gin.REQUIRED,
                 max_steps=gin.REQUIRED,
                 device=None):
        self.checkpoint_path = checkpoint_path
        self.max_steps = max_steps
        self.device = device or hf.choose_device()
        self.act_rep = ActionRepresentationPointer()
        self.tokenizer = IntPolicyTokenizerPointer(
            model_max_length=512,
            padding_side='right',
            pad_token=infix.PADDING_LEXEME
        )
        self.env = TheoremProverEnv()
        self.model = None
        self.prediction_counter = 0

    def reset_counter(self):
        self.prediction_counter = 0

    def read_counter(self):
        return self.prediction_counter

    def construct_networks(self):
        self.model = MBartForConditionalGeneration.from_pretrained(
            self.checkpoint_path
        ).to(self.device)

    def predict_action_strs(self, state_target_formulas):
        self.prediction_counter += 1
        dataset = GoalDataset.from_policy_input(
            state_target_formulas, self.tokenizer, padding=True,
            max_length=512,
        )
        inputs = [
            hf_generate.GenerationInput(
                input_ids=entry['input_ids'],
                attention_mask=entry['attention_mask']
            )
            for entry in dataset
        ]
        model_outputs, _ = hf_generate.generate_sequences(
            model=self.model,
            inputs=inputs,
            num_beams=5,
            num_return_sequences=1,
            max_length=512
        )
        padded_action_strs = self.tokenizer.batch_decode(model_outputs.squeeze(1))
        action_strs = [
            action_str.rstrip(infix.PADDING_LEXEME)
            for action_str in padded_action_strs
        ]
        assert len(action_strs) == len(state_target_formulas)
        return action_strs

    @staticmethod
    def _str_to_action(action_str, mask_to_entity):
        decoded_action = decode_prediction(action_str)
        if decoded_action is None:
            return None

        full_action = [decoded_action[0]]
        for entity_str in decoded_action[1:]:
            if entity_str in mask_to_entity:
                full_action.append(mask_to_entity[entity_str])
            else:
                return None
        return full_action

    def predict_actions(self, states_subgoals):
        state_target_formulas = []
        masks_to_entities = []
        for proof_state, subgoal_str in states_subgoals:
            state_target_formulas.append(
                self.act_rep.proof_states_to_policy_input_formula(
                    current_state=proof_state,
                    destination_state=subgoal_str
                )
            )
            _, mask_to_entity = generate_masks_for_logic_statement(
                int_utils.get_objective(proof_state)
            )
            masks_to_entities.append(mask_to_entity)

        actions_str = self.predict_action_strs(state_target_formulas)

        return [
            # There are Nones for incorrect actions.
            self._str_to_action(action_str, mask_to_entity)
            for action_str, mask_to_entity in zip(actions_str, masks_to_entities)
        ]

    def is_solved(self, state):
        sides = state.split('=')
        if len(sides) != 2:
            print(f'Invalid number of sides in the equation: {state}')
            return False
        l, r = sides
        return l == r

    def reach_subgoals(self, proof_state, subgoal_strs, force_step_limit=None):
        results = [None] * len(subgoal_strs)
        if int_utils.count_objectives(proof_state) > 1:
            # Multi objectives are not supported.
            return results

        pursuits = {
            idx: SubgoalPursuitData(
                subgoal_str=subgoal_str,
                current_state=deepcopy(proof_state),
                actions=[],
                intermediate_states=[],
                seen_states=set(logic_statement_to_seq_string(
                    int_utils.get_objective(proof_state)
                )),
                is_finished=False,
                subgoal_reached=False,
                done=False,
            )
            for idx, subgoal_str in enumerate(subgoal_strs)
        }

        step_limit = self.max_steps if force_step_limit is None else force_step_limit

        steps_taken = 0
        while len(pursuits) > 0 and steps_taken < step_limit:
            states_subgoals = [
                (pursuit.current_state, pursuit.subgoal_str)
                for pursuit in pursuits.values()
            ]
            actions = self.predict_actions(states_subgoals)
            assert len(actions) == len(pursuits)

            for pursuit, action in zip(pursuits.values(), actions):
                if action is None:
                    pursuit.is_finished = True
                    continue

                self.env.load_problem_step(pursuit.current_state)
                new_state, _, done, _ = self.env.step(action)
                if int_utils.count_objectives(new_state) > 1:
                    pursuit.is_finished = True
                    continue
                new_state_str = logic_statement_to_seq_string(
                    int_utils.get_objective(new_state)
                )
                if new_state_str in pursuit.seen_states:
                    pursuit.is_finished = True
                    continue
                pursuit.seen_states.add(new_state_str)

                pursuit.actions.append(action)
                pursuit.current_state = deepcopy(new_state)
                pursuit.intermediate_states.append(deepcopy(new_state))
                # Only keep track of the last 'done' value. We care only about
                # reaching the subgoal. We don't break even if the proof is completed.
                pursuit.done = done

                if new_state_str == pursuit.subgoal_str:
                    pursuit.subgoal_reached = pursuit.is_finished = True

            next_round_pursuits = {}
            for idx, pursuit in pursuits.items():
                if pursuit.subgoal_reached:
                    results[idx] = SubgoalPath(
                        actions=pursuit.actions,
                        intermediate_states=pursuit.intermediate_states,
                        subgoal_state=pursuit.current_state,
                        done=pursuit.done,
                    )
                elif not pursuit.is_finished:
                    next_round_pursuits[idx] = pursuit
                # Otherwise we drop the pursuit.
            pursuits = next_round_pursuits

            steps_taken += 1

        return results

