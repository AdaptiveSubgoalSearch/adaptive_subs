from goal_builders.shooting.goal_builder import GoalBuilderForShooting


class GoalBuilderForShootingRubik(GoalBuilderForShooting):
    def __init__(self, goal_generator, conditional_policy):
        self._goal_generator = goal_generator
        self._conditional_policy = conditional_policy

        self._goal_generator.construct_networks()
        self._conditional_policy.construct_networks()

    def build_goal(self, state):
        goal_prob_list = self._goal_generator.generate_subgoals(state)
        if not goal_prob_list:
            return None
        [raw_subgoal] = goal_prob_list
        goal_str = '?' + raw_subgoal[2:]
        reached, path, done, current_proof_state = self._conditional_policy.reach_subgoal(state, goal_str)
        # [goal_path] = self._conditional_policy.reach_subgoal(state, goal_str)
        if not reached:
            return None
        # assert len(goal_path.actions) == len(goal_path.intermediate_states)
        return (
            current_proof_state,
            done,
            list(zip(path, [current_proof_state] * len(path)))
        )
