class GoalBuilderForShooting:
    def build_goal(self, state):
        """For every given state randomly samples a subgoal.

        Returns:
            Tuples (next_subgoal_state, done, arbitrary_info_you_want)
            or None if incorrect subgoal was sampled.
        """
