import collections

import joblib

from supervised.int.gen_subgoal_data import generate_problems
import numpy as np

from supervised.int.representation.action_representation_pointer import ActionRepresentationPointer


SubgoalVerifiedPath = collections.namedtuple('SubgoalVerifiedPath', [
    'actions',
    'intermediate_states',
    'subgoal_state',
    'done',
    'verified',
])


class GoalBuilderINT:
    def __init__(self,
                 generator_class=None,
                 policy_class=None,
                 model_id=None,
                 gather_data_for_verificator=False,
                 verificator_class=None,
                 verificator_thresholds=[0, 1],  # [negative, positive]
                 verificator_stats_thresholds=[],
                 max_policy_steps=None
                 ):
        self.model_id = model_id
        if model_id is None:
            self.generator = generator_class()
        else:
            self.generator = generator_class(generator_checkpoint_path=model_id)

        self.max_policy_steps = max_policy_steps
        self.policy = policy_class(max_steps=max_policy_steps)

        if verificator_class is None:
            self.verificator = None
        else:
            self.verificator = verificator_class()

        self.gather_data_for_verificator = gather_data_for_verificator
        self.positive_subgoals = []
        self.negative_subgoals = []
        self.act_rep = ActionRepresentationPointer()

        assert np.array_equal(verificator_stats_thresholds,
                              sorted(verificator_stats_thresholds))  # thresholds have to be increasing
        self.verificator_stats_thresholds = verificator_stats_thresholds
        self.verifications = np.zeros((2, len(verificator_stats_thresholds) + 1))

        assert len(verificator_thresholds) == 2
        self.verificator_thresholds = verificator_thresholds

        self.calls_generator = 0
        self.calls_policy = 0
        self.calls_verificator = 0

    def reset_counter(self):
        self.generator.reset_counter()
        self.policy.reset_counter()

    def read_counter(self):
        return {
            'generator': self.generator.read_counter(),
            'policy': self.policy.read_counter()
        }

    def construct_networks(self):
        self.generator.construct_networks()
        self.policy.construct_networks()
        if self.verificator is not None:
            self.verificator.construct_networks()

    def log_verification(self, state, subgoal, reached):
        prob = self.verificator.predict_reachability(self.make_verificator_input(state, subgoal))

        group = 0
        for th in self.verificator_stats_thresholds:
            if prob > th:
                group += 1
            else:
                break
        self.verifications[int(reached), group] += 1

        return prob

    def verificator_probability(self, state, subgoal):
        if self.verificator is None:
            return None

        self.calls_verificator += 1
        return self.verificator.predict_reachability(self.make_verificator_input(state, subgoal))

    def save_verificator_data(self):
        if self.gather_data_for_verificator:
            escaped_id = self.model_id.replace('/', '|')
            joblib.dump([self.positive_subgoals, self.negative_subgoals],
                        f'output/vdata_builder_{escaped_id}_rnd{np.random.randint(1e10)}', compress=5)

    def make_verificator_input(self, state, subgoal):
        return self.act_rep.proof_states_to_policy_input_formula(state, subgoal)

    def build_goals(self, current_state):
        real_cost = 0

        self.calls_generator += self.generator.num_beams
        raw_subgoals = self.generator.generate_subgoals(current_state)
        subgoal_strs = [raw_subgoal[0] for raw_subgoal in raw_subgoals]

        if self.gather_data_for_verificator:
            results = self.policy.reach_subgoals(current_state, subgoal_strs)

            for subgoal, result in zip(subgoal_strs, results):
                if result is None:
                    self.negative_subgoals.append(self.make_verificator_input(current_state, subgoal))
                else:
                    self.positive_subgoals.append(self.make_verificator_input(current_state, subgoal))

        if len(self.verificator_stats_thresholds) > 0:
            # Log verification statistics
            results = self.policy.reach_subgoals(current_state, subgoal_strs)
            for subgoal, result in zip(subgoal_strs, results):
                self.log_verification(current_state, subgoal, result is not None)

        subgoals_to_verify = []

        for subgoal in subgoal_strs:
            verificator_prob = self.verificator_probability(current_state, subgoal)

            if verificator_prob is not None and verificator_prob > self.verificator_thresholds[1]:
                # NOTE Currently impossible, requires recreating environment state from string representation
                real_cost += 1
                assert False
            elif verificator_prob is not None and verificator_prob < self.verificator_thresholds[0]:
                # unreachable
                real_cost += 1
                continue
            else:
                # verificator is uncertain
                subgoals_to_verify.append(subgoal)

        results = self.policy.reach_subgoals(current_state, subgoals_to_verify)
        verified_subgoals = []

        for subgoal, result in zip(subgoals_to_verify, results):
            if result is None:
                self.calls_policy += self.max_policy_steps * 5  # num_beams for policy
                real_cost += self.max_policy_steps * 5  # num_beams for policy
                continue

            # reached

            verified_subgoals.append(SubgoalVerifiedPath(
                actions=result.actions,
                intermediate_states=result.intermediate_states,
                subgoal_state=result.subgoal_state,
                done=result.done,
                verified=False,
            ))

            self.calls_policy += len(result.actions) * 5
            real_cost += len(result.actions) * 5

        return verified_subgoals, real_cost
