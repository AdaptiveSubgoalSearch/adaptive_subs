import numpy as np

from envs import Sokoban
from goal_builders.sokoban.goal_builder import GoalBuilder
from goal_builders.sokoban.goal_builder_node import GoalBuilderNode
from utils.utils_sokoban import HashableNumpyArray


class PolicyGoalBuilderSokoban(GoalBuilder):
    def __init__(self, policy_class):
        self.policy = policy_class()
        self.core_env = Sokoban()
    
    def construct_networks(self):
        self.policy.construct_networks()

    def build_goals(
        self,
        input,
        max_radius,
        total_confidence_level,
        internal_confidence_level,
        max_goals,
        reverse_order
    ):
        root = GoalBuilderNode(input, np.copy(input), 1, 0, False, 0, 0, None)
        goals = []
        hashed_goals = {HashableNumpyArray(input)}
        accumulated_confidence = 0

        actions = self.policy.predict_actions(input)
        sorted_indexes = [i[0] for i in sorted(enumerate(actions), key=lambda x:x[1], reverse=reverse_order)]

        for action in sorted_indexes:
            self.core_env.restore_full_state_from_np_array_version(input)
            obs, _, _, _ = self.core_env.step(action)

            if HashableNumpyArray(obs) not in hashed_goals:
                new_goal = GoalBuilderNode(input, obs, actions[action], 0, True, len(goals), 1, root)
                new_goal.add_path_info(tuple([action]))
                root.children.append(new_goal)
                goals.append(new_goal)
                hashed_goals.add(HashableNumpyArray(obs))
                accumulated_confidence += actions[action]

            if accumulated_confidence > total_confidence_level:
                break

        return goals, 0, 0, 0, 0, 0, 0, 0, 0
