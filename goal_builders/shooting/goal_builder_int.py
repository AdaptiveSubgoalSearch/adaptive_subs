from goal_builders.shooting.goal_builder import GoalBuilderForShooting


class GoalBuilderForShootingINT(GoalBuilderForShooting):
    def __init__(self, goal_generator, conditional_policy):
        self._goal_generator = goal_generator
        self._conditional_policy = conditional_policy

        self._goal_generator.construct_networks()
        self._conditional_policy.construct_networks()

    def build_goal(self, state):
        goal_prob_list = self._goal_generator.generate_subgoals(state)
        if not goal_prob_list:
            return None
        [(goal_str, _)] = goal_prob_list
        [goal_path] = self._conditional_policy.reach_subgoals(state, [goal_str])
        if not goal_path:
            return None
        assert len(goal_path.actions) == len(goal_path.intermediate_states)
        return (
            goal_path.subgoal_state,
            goal_path.done,
            list(zip(goal_path.actions, goal_path.intermediate_states))
        )

    def reset_counter(self):
        self._goal_generator.reset_counter()
        self._conditional_policy.reset_counter()

    def read_counter(self):
        return {
            'generator': self._goal_generator.read_counter(),
            'policy': self._conditional_policy.read_counter(),
        }
