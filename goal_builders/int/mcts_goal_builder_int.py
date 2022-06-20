import collections

from alpacka.agents.tree_search import GoalBuilder
from alpacka.agents.tree_search import ChildInfo
from metric_logging import log_text
from supervised.int.representation import infix


GoalIntAction = collections.namedtuple('GoalIntActions', [
    'action', 'resulting_state'
])


class MCTSGoalBuilderInt(GoalBuilder):
    def __init__(self, goal_generator, conditional_policy):
        self._goal_generator = goal_generator
        self._conditional_policy = conditional_policy

        self._goal_generator.construct_networks()
        self._conditional_policy.construct_networks()

    def reset_counter(self):
        self._goal_generator.reset_counter()
        self._conditional_policy.reset_counter()

    def read_counter(self):
        return {
            'generator': self._goal_generator.read_counter(),
            'policy': self._conditional_policy.read_counter(),
        }

    def build_goals(self, state):
        subgoal_strs, subgoal_probs = zip(*self._goal_generator.generate_subgoals(state))

        subgoal_paths = self._conditional_policy.reach_subgoals(state, subgoal_strs)
        if all(result is None for result in subgoal_paths):
            log_text(
                'policy_dead_ends',
                f'For state {infix.InfixRepresentation.proof_state_to_input_formula(state)} '
                'none of following subgoals was reached: '
                + '; '.join(subgoal_strs)
            )
        assert len(subgoal_strs) == len(subgoal_paths)

        result = []
        for (subgoal_path, subgoal_prob) in zip(subgoal_paths, subgoal_probs):
            if subgoal_path is None:
                continue
            assert len(subgoal_path.actions) == len(subgoal_path.intermediate_states)
            result.append((
                subgoal_path.subgoal_state,
                ChildInfo(
                    action_list=[
                        GoalIntAction(action=action, resulting_state=state)
                        for action, state in zip(
                            subgoal_path.actions, subgoal_path.intermediate_states
                        )
                    ],
                    reward=1 if subgoal_path.done else 0,
                    done=subgoal_path.done
                ),
                subgoal_prob
            ))
        return result
