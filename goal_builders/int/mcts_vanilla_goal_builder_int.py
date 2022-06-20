import gin

from alpacka.agents.tree_search import GoalBuilder
from alpacka.agents.tree_search import ChildInfo

from envs.int.theorem_prover_env import TheoremProverEnv
from goal_builders.int.mcts_goal_builder_int import GoalIntAction
from policies.int.vanilla_policy import VanillaPolicyINT


class MCTSVanillaGoalBuilderInt(GoalBuilder):
    def __init__(self, checkpoint_path, num_beams, num_goals):
        self._env = TheoremProverEnv()

        self._num_beams = num_beams
        self._num_goals = num_goals

        self._vanilla_policy = VanillaPolicyINT(
            checkpoint_path,
            num_beams=self._num_beams,
            num_return_sequences=self._num_goals,
            max_steps_allowed=0,  # Irrelevant in the case of goal builder.
        )
        self._vanilla_policy.construct_networks()

    def reset_counter(self):
        self._vanilla_policy.reset_counter()

    def read_counter(self):
        return {'policy': self._vanilla_policy.read_counter()}

    def build_goals(self, state):
        actions_states_probs, _ = self._vanilla_policy.predict_actions(
            proof_state=state,
        )

        result = []
        for action, state, prob in actions_states_probs:
            self._env.load_problem_step(state)
            new_state, _, done, _ = self._env.step(action)
            if len(new_state['observation']['objectives']) > 1:
                # We don't allow the policy to introduce more than one objective.
                continue

            result.append((
                new_state,
                ChildInfo(
                    action_list=[
                        GoalIntAction(action=action, resulting_state=new_state)
                    ],
                    reward=1 if done else 0,
                    done=done
                ),
                prob
            ))
        return result
