# from supervised.int.gen_subgoal_data import generate_problems
import joblib
import numpy as np

from supervised.rubik.gen_rubik_data import encode_policy_subgoal
from supervised.rubik.rubik_solver_utils import generate_problems_rubik


class GoalBuilderRubik:
    def __init__(self,
                 generator_class=None,
                 generator_id=None,
                 policy_class=None,
                 gather_data_for_verificator=False,
                 verificator_class=None,
                 verificator_thresholds=[0, 1],  # [negative, positive]
                 verificator_stats_thresholds=[],
                 max_policy_steps=None,
                 add_distance_token=False):
        if generator_id is None:
            self.generator = generator_class()
        else:
            self.generator = generator_class(generator_checkpoint_path=generator_id,
                                             add_distance_token=max_policy_steps if add_distance_token else None)
        self.generator_id = generator_id
        self.policy = policy_class(max_steps=max_policy_steps)

        if verificator_class is None:
            self.verificator = None
        else:
            self.verificator = verificator_class()

        self.gather_data_for_verificator = gather_data_for_verificator
        self.positive_subgoals = []
        self.negative_subgoals = []

        assert np.array_equal(verificator_stats_thresholds, sorted(verificator_stats_thresholds))  # thresholds have to be increasing
        self.verificator_stats_thresholds = verificator_stats_thresholds
        self.verifications = np.zeros((2, len(verificator_stats_thresholds) + 1))

        self.verificator_thresholds = verificator_thresholds
        assert len(verificator_thresholds) == 2
        if verificator_thresholds[0] is not None and verificator_thresholds[1] is not None:
            assert verificator_thresholds[0] <= verificator_thresholds[1]

        self.subgoals_skipped = 0
        self.subgoals_tested = 0

        self.calls_generator = 0
        self.calls_policy = 0
        self.calls_verificator = 0

    def construct_networks(self):
        self.generator.construct_networks()
        self.policy.construct_networks()
        if self.verificator is not None:
            self.verificator.construct_networks()

    def log_verification(self, state, subgoal, prob):
        if self.verificator is None:
            return

        reached, _, _, _ = self.policy.reach_subgoal(state, subgoal)
        print(f'Reached: {reached}, Verificator answer: {prob}')

        group = 0
        for th in self.verificator_stats_thresholds:
            if prob > th:
                group += 1
            else:
                break
        self.verifications[int(reached), group] += 1

    def verificator_probability(self, state, subgoal):
        if self.verificator is None:
            return None

        self.calls_verificator += 1
        prob = self.verificator.predict_reachability(state, subgoal)

        if len(self.verificator_stats_thresholds) > 0:
            self.log_verification(state, subgoal, prob)

        return prob

    def save_verificator_data(self):
        if self.gather_data_for_verificator:
            escaped_id = self.generator_id.replace('/', '|')
            joblib.dump([self.positive_subgoals, self.negative_subgoals],
                        f'output/vdata_builder_{escaped_id}_rnd{np.random.randint(10000000, 99999999)}', compress=5)

    def build_goals(self, current_state):
        raw_subgoals = self.generator.generate_subgoals(current_state)
        self.calls_generator += self.generator.num_beams
        verifed_subgoals = []
        real_cost = 0

        for raw_subgoal in raw_subgoals:
            # raw subgoal starts with '$@', should be changed to '?'
            raw_subgoal = '?' + raw_subgoal[2:]

            self.subgoals_tested += 1
            verificator_prob = self.verificator_probability(current_state, raw_subgoal)

            if verificator_prob is not None and verificator_prob > self.verificator_thresholds[1]:
                path = [-1]
                reached = True
                done = self.policy.is_solved(raw_subgoal)
                subgoal_real_cost = 1  # only verificator is used
                self.subgoals_skipped += 1
                step_calls_policy = 0
            elif verificator_prob is not None and verificator_prob < self.verificator_thresholds[0]:
                reached = False
                path = []
                done = False
                subgoal_real_cost = 1  # only verificator is used
                self.subgoals_skipped += 1
                step_calls_policy = 0
            else:  # verificator is uncertain
                policy_steps = [0]
                reached, path, done, current_proof_state = self.policy.reach_subgoal(current_state, raw_subgoal, steps_counter=policy_steps)
                subgoal_real_cost = policy_steps[0]
                step_calls_policy = policy_steps[0]
                verificator_prob = None

            real_cost += subgoal_real_cost
            subgoal_data = (raw_subgoal, path, done, verificator_prob)
            self.calls_policy += step_calls_policy

            if reached:
                verifed_subgoals.append(subgoal_data)
                if self.gather_data_for_verificator:
                    self.positive_subgoals.append(encode_policy_subgoal(current_state, raw_subgoal))
            else:
                if self.gather_data_for_verificator:
                    self.negative_subgoals.append(encode_policy_subgoal(current_state, raw_subgoal))
            if done:
                return verifed_subgoals, subgoal_data, real_cost
        return verifed_subgoals, None, real_cost


def goal_builder_test():
    PROOFS = 3
    tmp = GoalBuilderRubik()
    tmp.construct_networks()
    example_problems = generate_problems_rubik(PROOFS)

    for i in range(PROOFS):
        print(
            '______________________________________START_______________________________________________________________')
        example_problem = example_problems[i]
        subgoals, done = tmp.build_goals(example_problem[0])
        print(f'Subgoals {len(subgoals)}: {subgoals}')
    print(
        '_________________________________________END____________________________________________________________ \n \n')

# goal_builder_test()
