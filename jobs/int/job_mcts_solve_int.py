import time

import gin

from jobs.core import Job
from metric_logging import log_scalar, log_scalar_metrics, MetricsAccumulator
from solvers.mcts import mcts_solve
from supervised.int import gen_subgoal_data
from supervised.int.representation import infix
from supervised.int.utils import print_proof_state
from third_party.INT.visualization import seq_parse
from value_estimators.int.value_estimator_int import ValueEstimatorINT


def print_solution(transitions):
    for transition in transitions:
        for action_state in transition.action_list:
            print(
                f'Action: {action_state.action[0]} | '
                f'{[seq_parse.entity_to_seq_string(x) for x in action_state.action[1:]]}'
            )
            state_str = infix.InfixRepresentation.proof_state_to_input_formula(
                action_state.resulting_state
            )
            print(f'State: {state_str}')
        subgoal_str = infix.InfixRepresentation.proof_state_to_input_formula(
            transition.next_observation
        )
        print(f'Subgoal: {subgoal_str}')


class JobMCTSSolveInt(Job):
    def __init__(
        self,
        goal_builder=gin.REQUIRED,
        value_estimator_checkpoint=gin.REQUIRED,
        n_proofs=3,
        time_limit=8,
        discount=0.99,
        skip_first_n_problems=0,
        n_shards=1,
        shard_id=0,
        budget_checkpoints=None,
    ):
        self._goal_builder = goal_builder
        self._value_estimator_checkpoint = value_estimator_checkpoint
        self._n_proofs = n_proofs
        self._time_limit = time_limit
        self._discount = discount
        self._skip_first_n_problems = skip_first_n_problems
        self._n_shards = n_shards
        self._shard_id = shard_id
        self._budget_checkpoints = budget_checkpoints

    def execute(self):
        value_estimator = ValueEstimatorINT(
            checkpoint_path=self._value_estimator_checkpoint
        )
        value_estimator.construct_networks()

        def value_function(states):
            negative_dists = value_estimator.evaluate(states)
            return [
                self._discount ** (-negative_dist)
                for negative_dist in negative_dists
            ]

        solved_metrics = MetricsAccumulator()

        for i in range(self._n_proofs):
            [problem] = gen_subgoal_data.generate_problems(n_proofs=1)
            if i < self._skip_first_n_problems or i % self._n_shards != self._shard_id:
                continue

            init_state = problem[0]
            del init_state['lemma']
            print(f'Problem {i}')
            print_proof_state(init_state)

            self._goal_builder.reset_counter()
            value_estimator.reset_counter()
            t_solve = time.time()
            episode = mcts_solve(
                goal_builder=self._goal_builder,
                value_function=value_function,
                init_state=init_state,
                time_limit=self._time_limit,
                discount=self._discount,
            )
            log_scalar('time_solve', i, time.time() - t_solve)
            log_scalar_metrics('predictions', i, self._goal_builder.read_counter())
            log_scalar_metrics('predictions', i, value_estimator.read_counter())

            print_solution(episode.transitions)
            log_scalar('solution', i, episode.solved)
            log_scalar(
                'solution/length', i,
                count_actions(episode.transitions) if episode.solved else -1
            )
            log_scalar(
                'solution/n_subgoals', i,
                count_subgoals(episode.transitions) if episode.solved else -1
            )
            solved_metrics.log_metric_to_average('rate', episode.solved)
            solved_metrics.log_metric_to_accumulate('problems', episode.solved)
            # if self._budget_checkpoints:
            #     for budget in self._budget_checkpoints:
            #         solved_metrics.log_metric_to_average(
            #             f'rate/{budget}_exp_nodes',
            #             int(solved and agent_info['inner_nodes'] <= budget)
            #         )
            #         solved_metrics.log_metric_to_average(
            #             f'rate/{budget}_nodes',
            #             int(solved and agent_info['nodes'] <= budget)
            #         )

            log_scalar_metrics('solved', i, solved_metrics.return_scalars())


def count_actions(transitions):
    return sum(
        len(transition.action_list)
        for transition in transitions
    )


def count_subgoals(transitions):
    return len(transitions)
