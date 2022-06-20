from envs.int.theorem_prover_env import TheoremProverEnv
from policies import VanillaPolicyINT
from policies.int.transformer_conditional_policy_pointer import SubgoalPursuitData, SubgoalPath
from supervised.int.gen_subgoal_data import generate_problems
from supervised.int.utils import count_objectives
from visualization.seq_parse import logic_statement_to_seq_string


class VanillaGoalBuilderINT:
    def __init__(self,
                 vanilla_policy_class
                 ):

        self.vanilla_policy = vanilla_policy_class()
        self.env = TheoremProverEnv()

    def reset_counter(self):
        self.vanilla_policy.reset_counter()

    def read_counter(self):
        return {'policy': self.vanilla_policy.read_counter()}

    def construct_networks(self):
        self.vanilla_policy.construct_networks()

    def build_goals(self, current_state):
        actions = self.vanilla_policy.predict_actions(current_state)
        seen_state_str = set()
        verifed_subgoals = []

        for action in actions[0]:
            state = action[1]
            executable_action = action[0]
            self.env.load_problem_step(state)
            new_state, _, done, _ = self.env.step(executable_action)
            new_state_str = logic_statement_to_seq_string(new_state['observation']['objectives'][0])
            if new_state_str not in seen_state_str and count_objectives(new_state) == 1:
                seen_state_str.add(new_state_str)
                new_subgoal_data = SubgoalPath([executable_action], [], new_state, done)
                verifed_subgoals.append(new_subgoal_data)
                if done:
                    return verifed_subgoals


        return verifed_subgoals