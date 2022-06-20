from envs import Sokoban
from goal_builders.sokoban.goal_builder import GoalBuilder
from goal_builders.sokoban.goal_builder_node import GoalBuilderNode
from utils.utils_sokoban import HashableNumpyArray


class TrivialGoalBuilderSokoban(GoalBuilder):
    def __init__(self):
        self.core_env = Sokoban()

    def build_goals(self, input, max_radius, total_confidence_level, _, max_goals, reverse_order):
        del max_radius
        del total_confidence_level
        del max_goals
        del reverse_order

        goals = []
        hashed_goals = {HashableNumpyArray(input)}

        for action in range(4):
            self.core_env.restore_full_state_from_np_array_version(input)
            obs, _, _, _ = self.core_env.step(action)

            if HashableNumpyArray(obs) not in hashed_goals:
                new_goal = GoalBuilderNode(input, obs, 1, None, True, len(goals), 0, None)
                new_goal.add_path_info(tuple([action]))
                goals.append(new_goal)
                hashed_goals.add(HashableNumpyArray(obs))

        return goals
