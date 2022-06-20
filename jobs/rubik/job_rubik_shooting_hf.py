import pprint
import time

import gin

from jobs.core import Job
from metric_logging import log_scalar, log_scalar_metrics, MetricsAccumulator, log_text
from solvers.shooting import random_shooting_solve
# from supervised.int import gen_subgoal_data
# from supervised.int.representation import infix
# from supervised.int.utils import print_proof_state, get_objective
# from third_party.INT.visualization import seq_parse
from supervised.rubik.rubik_solver_utils import generate_problems_rubik
from goal_builders import GoalBuilderForShootingRubik


def print_solution(transitions):
    for transition in transitions:
        for action, state in transition.info:
            print(f'Action: {action} | ()')
            print(f'State: {state}')
        subgoal_str = transition.next_state
        print(f'Subgoal: {subgoal_str}')


class JobShootingSolveRubik(Job):
    def __init__(
        self,
        goal_builder=gin.REQUIRED,
        n_proofs=3,
        time_limit=8,
        n_trajectories=3,
        budget_checkpoints=(1, 2, 3, 100),
        n_shards=1,
        shard_id=0,
        skip_first_n_problems=0
    ):
        self._goal_builder = goal_builder
        self._n_proofs = n_proofs
        self._time_limit = time_limit
        self._n_trajectories = n_trajectories
        self._budget_checkpoints = budget_checkpoints

        self._n_shards = n_shards
        self._shard_id = shard_id
        self._skip_first_n_problems = skip_first_n_problems

    def execute(self):
        solved_metrics = MetricsAccumulator()

        for i in range(self._n_proofs):
            [problem] = generate_problems_rubik(n_problems=1)
            if i < self._skip_first_n_problems or i % self._n_shards != self._shard_id:
                continue

            init_state = problem[0]
            print(f'Problem {i}')
            print('proof state', init_state)

            t_solve = time.time()
            episode = random_shooting_solve(
                init_state=init_state,
                goal_builder=self._goal_builder,
                time_limit=self._time_limit,
                n_trajectories=self._n_trajectories,
                state_to_str=lambda state: state
            )
            log_scalar('time_solve', i, time.time() - t_solve)
            if episode.done:
                print_solution(episode.transitions)
            log_scalar('solution', i, episode.done)
            log_scalar('problems/tested', i, i+1)
            log_scalar(
                'solution/length', i,
                count_actions(episode.transitions) if episode.done else -1
            )
            log_scalar(
                'solution/n_subgoals', i,
                count_subgoals(episode.transitions) if episode.done else -1
            )
            solved_metrics.log_metric_to_average('rate', episode.done)
            solved_metrics.log_metric_to_accumulate('problems', episode.done)
            if self._budget_checkpoints:
                for budget in self._budget_checkpoints:
                    solved_metrics.log_metric_to_average(
                        f'rate/{budget}_trials',
                        int(episode.done and episode.n_trials <= budget)
                    )
            log_scalar_metrics('solved', i, solved_metrics.return_scalars())
            log_scalar(
                'different_trajectories', i,
                len(episode.trajectory_counts)
            )
            log_scalar(
                'different_trajectories_rate', i,
                len(episode.trajectory_counts) / episode.n_trials
            )
            trajectories = list(reversed(sorted([
                (count, trajectory)
                for trajectory, count in episode.trajectory_counts.items()
            ])))
            pprint.pprint(trajectories)
            log_text(
                'trajectories_distribution',
                str(list(map(lambda x: x[0], trajectories))),
                show_on_screen=True,
            )


def count_actions(transitions):
    return sum(
        len(transition.info)
        for transition in transitions
    )


def count_subgoals(transitions):
    return len(transitions)
