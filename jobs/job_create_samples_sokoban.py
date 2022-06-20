import time

from envs import Sokoban
from goal_builders import BFSGoalBuilderSokoban
from jobs.core import Job
from supervised import DataCreatorSokoban
from utils.general_utils import readable_num
from utils.utils_sokoban import draw_and_log, many_states_to_fig
from metric_logging import log_image, log_scalar

class JobCreateSamplesSokoban(Job):
    def __init__(self,
                 goal_builder_class,
                 n_tests,
                 max_radius,
                 total_confidence_level,
                 internal_confidence_level,
                 max_goals,
                 only_correct=True
                 ):

        self.goal_builder = goal_builder_class()
        self.n_tests = n_tests
        self.max_radius = max_radius
        self.total_confidence_level = total_confidence_level
        self.internal_confidence_level = internal_confidence_level
        self.max_goals = max_goals
        self.only_correct = only_correct

        self.data_creator = DataCreatorSokoban()
        self.core_env = Sokoban()

    def execute(self):
        total_time_start = time.time()
        self.goal_builder.construct_networks()
        for test_num in range(self.n_tests):
            input = self.core_env.reset()
            raw_goals = self.goal_builder._generate_goals(self.internal_confidence_level, input)
            good_goals = self.goal_builder.build_goals(input, self.max_radius, self.total_confidence_level, self.internal_confidence_level, self.max_goals, True)
            good_goals_set = {goal.hashed_goal for goal in good_goals}

            all_goals = [input]
            titles = ['input']

            if self.only_correct:
                raw_goals = good_goals

            for goal in raw_goals[:self.max_goals + 1]:
                all_goals.append(goal.goal_state)
                if goal.hashed_goal in good_goals_set:
                    titles.append(f'ok p = {int(10000 * goal.p) / 10000}')
                else:
                    titles.append(f'wrong p = {int(10000 * goal.p) / 10000}')

            if len(all_goals) > 1:
                fig = many_states_to_fig(all_goals, titles)
            else:
                fig = draw_and_log(all_goals[0], 'input (no samples)', 'samples', test_num)

            collected_total_p = sum([goal.p for goal in raw_goals])
            collected_good_p = sum([goal.p for goal in good_goals])


            log_image('samples',test_num, fig)
            log_scalar('collected_good_p', test_num, collected_good_p)
            log_scalar('collected_total_p', test_num, collected_total_p)
            log_scalar('collected_wrong_p', test_num, collected_total_p - collected_good_p)
            log_scalar('n_goals', test_num, len(raw_goals))
            finished_time = readable_num(time.time() - total_time_start)
            return finished_time