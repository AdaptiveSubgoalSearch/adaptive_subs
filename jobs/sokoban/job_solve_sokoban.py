import sys
import time
import numpy as np

from joblib import (
    delayed,
    Parallel,
    load
)

from envs import Sokoban
from graph_tracer import GraphTracerSokoban
from jobs.core import Job
from metric_logging import (
    log_scalar,
    log_scalar_metrics,
    log_text,
    MetricsAccumulator,
)
from utils.general_utils import readable_num
from utils.utils_sokoban import draw_and_log
from third_party.keras_pickable import make_keras_picklable


class JobSolveSokoban(Job):
    def __init__(
        self,
        solver_class,
        n_jobs,
        n_parallel_workers,
        batch_size,
        initial_batch=None,
        budget_checkpoints=None,
        boards_dataset=None,
        env_seed=None,
        log_solutions_limit=100,
        boards_indices=None,
        use_graph_tracer=False,
        recursion_limit=None
    ):
        self.solver_class = solver_class
        self.n_jobs = n_jobs
        self.n_parallel_workers = n_parallel_workers
        self.batch_size = batch_size
        self.initial_batch = initial_batch
        self.budget_checkpoints = budget_checkpoints
        self.boards_dataset = boards_dataset
        self.log_solutions_limit = log_solutions_limit
        self.boards_indices = boards_indices

        self.core_env = Sokoban(seed=env_seed)
        self.solved_stats = MetricsAccumulator()
        self.experiment_stats = MetricsAccumulator()
        self.subgoal_stats = MetricsAccumulator()

        self.logged_solutions = 0

        self.graph_tracer = None
        self.use_graph_tracer = use_graph_tracer
        if self.use_graph_tracer:
            self.graph_tracer = GraphTracerSokoban()
        
        self.recursion_limit = recursion_limit

    def execute(self):
        total_time_start = time.time()
        make_keras_picklable()
        if self.boards_dataset is not None:
            with open(self.boards_dataset, 'rb') as handle:
                boards_to_solve = load(handle)
            if self.boards_indices is not None:
                boards_to_solve_list = [boards_to_solve[i] for i in self.boards_indices]
                boards_to_solve = boards_to_solve_list
            if self.n_jobs < len(boards_to_solve):
                boards_to_solve = boards_to_solve[:self.n_jobs]
            if self.n_jobs > len(boards_to_solve):
                self.n_jobs = len(boards_to_solve)
        else:
            boards_to_solve = [self.core_env.reset() for _ in range(self.n_jobs)]

        solver = self.solver_class()
        jobs_done = 0
        jobs_to_do = self.n_jobs
        batch_num = 0

        while jobs_to_do > 0:
            batch_limit = self.batch_size
            if self.initial_batch is not None and len(self.initial_batch) > 0:
                batch_limit = self.initial_batch[0]
                self.initial_batch = self.initial_batch[1:]

            jobs_in_batch = min(jobs_to_do, self.batch_size, batch_limit)
            boards_to_solve_in_batch = boards_to_solve[jobs_done:jobs_done + jobs_in_batch]
            self.print_batch_header(batch_num)

            results = Parallel(n_jobs=self.n_parallel_workers, verbose=100)(
                delayed(solve_board)(solver, input_board, self.use_graph_tracer, self.recursion_limit)
                for input_board in boards_to_solve_in_batch
            )

            self.print_solution_header(batch_num, time.time() - total_time_start)
            self.log_results(results, jobs_done)
            jobs_done += jobs_in_batch
            jobs_to_do -= jobs_in_batch
            batch_num += 1

            if self.graph_tracer is not None:
                for result in results:
                    tree_nodes = result['additional_info']['tree_nodes']
                    tree_edges = result['additional_info']['tree_edges']
                    tree_extra_edges = result['additional_info']['tree_extra_edges']
                    solution = result['solution']
                    self.graph_tracer.draw_graph(solution, tree_nodes, tree_edges, tree_extra_edges)

        for metric, value in self.solved_stats.return_scalars().items():
            log_text('summary', f'{metric},  {value}', show_on_screen=True)

        finished_time = readable_num(time.time() - total_time_start)
        log_text('summary', f'Finished time , {finished_time}', show_on_screen=True)

        return self.solved_stats, finished_time
    
    def print_batch_header(self, batch_num):
        all_batches = self.n_jobs // self.batch_size
        print(
            '============================ Batch {:>4}  out  of  {:>4} ============================'.
            format(batch_num + 1, all_batches)
        )

    def print_solution_header(self, batch_num, execution_time):
        print(
            '========================= Solved Batch {:>4}  in  {:>4} s ========================='.
            format(batch_num + 1, readable_num(execution_time))
        )

    def log_results(self, results, step):
        n_logs = len(results)

        for log_num, result in enumerate(results):
            log_scalar_metrics('tree', step + log_num, result['tree_metrics'])

            if self.logged_solutions < self.log_solutions_limit:
                self.log_solution(result['solution'], result['input_board'], step + log_num)

            solved = result['solution'] is not None
            self.experiment_stats.log_metric_to_accumulate('tested', 1)
            log_scalar_metrics('boards', step + log_num, self.experiment_stats.return_scalars())


            # Verificator metrics
            self.solved_stats.log_metric_to_accumulate('verificator_failed', result['tree_metrics']['verificator_failed'])
            self.solved_stats.log_metric_to_accumulate('verificator_succeed', result['tree_metrics']['verificator_succeed'])
            self.solved_stats.log_metric_to_accumulate('verificator_certain_number', result['tree_metrics']['verificator_certain_number'])
            self.solved_stats.log_metric_to_accumulate('verificator_verified_later_number', result['tree_metrics']['verificator_verified_later_number'])
            self.solved_stats.log_metric_to_accumulate('ver_calls', result['tree_metrics']['ver_calls'])
            self.solved_stats.log_metric_to_accumulate('cllp_calls', result['tree_metrics']['cllp_calls'])
            self.solved_stats.log_metric_to_accumulate('ver_samples_in_calls', result['tree_metrics']['ver_samples_in_calls'])
            self.solved_stats.log_metric_to_accumulate('cllp_samples_in_calls', result['tree_metrics']['cllp_samples_in_calls'])
            self.solved_stats.log_metric_to_accumulate('value_calls', result['tree_metrics']['value_calls'])
            self.solved_stats.log_metric_to_accumulate('subgoal_gen_calls', result['tree_metrics']['subgoal_gen_calls'])

            if result['tree_metrics']['verificator_first_fail_goal_index'] > 0:
                self.solved_stats.log_metric_to_average('verificator_first_fail_goal_index', result['tree_metrics']['verificator_first_fail_goal_index'])

            # Computations metrics
            self.solved_stats.log_metric_to_accumulate('total_nodes_with_computations_in_graph', result['tree_metrics']['total_nodes_with_computations_in_graph'])

            # Solver metrics
            if solved:
                self.solved_stats.log_metric_to_average('rate/all', 1)
                self.solved_stats.log_metric_to_accumulate('boards', 1)
                log_scalar('solution', step + log_num, 1)
                log_scalar('solution/length', step + log_num, len(result['trajectory_actions']))
                trajectory_actions = [str(action) for action in result['trajectory_actions']]
                trajectory = ', '.join(trajectory_actions)
                log_text('trajectory_actions', f'{step + log_num}: {trajectory}', False)
                log_scalar('solution/n_subgoals', step + log_num, len(result['solution']))
            else:
                self.solved_stats.log_metric_to_average('rate/all', 0)
                self.solved_stats.log_metric_to_accumulate('boards', 0)
                log_scalar('solution', step + log_num, 0)
                log_scalar('solution/length', step + log_num, -1)
                log_text('trajectory_actions', f'{step + log_num}: unsolved', False)
                log_scalar('solution/n_subgoals', step + log_num, -1)

            if self.budget_checkpoints is not None:
                for budget in self.budget_checkpoints:
                    if result['tree_metrics']['expanded_nodes'] <= budget and solved:
                        self.solved_stats.log_metric_to_average(f'rate/{budget}_exp_nodes', 1)
                    else:
                        self.solved_stats.log_metric_to_average(f'rate/{budget}_exp_nodes', 0)

                    if result['tree_metrics']['nodes'] <= budget and solved:
                        self.solved_stats.log_metric_to_average(f'rate/{budget}_nodes', 1)
                    else:
                        self.solved_stats.log_metric_to_average(f'rate/{budget}_nodes', 0)

                    if result['tree_metrics']['total_nodes_with_computations_in_graph'] <= budget and solved:
                        self.solved_stats.log_metric_to_average(f'rate/{budget}_total_nodes_with_computations_in_graph', 1)
                    else:
                        self.solved_stats.log_metric_to_average(f'rate/{budget}_total_nodes_with_computations_in_graph', 0)

        log_scalar_metrics('solved', step + n_logs - 1, self.solved_stats.return_scalars())


        for metric, value in self.subgoal_stats.return_scalars().items():
            log_scalar(metric, step + n_logs - 1, value)

    def log_solution(self, solution, input_board, step):
        self.logged_solutions += 1

        if solution is None:
            draw_and_log(input_board, 'input (unsolved)', f'episode_{step}', 0)
        else:
            draw_and_log(input_board, 'input (solved)', f'episode_{step}', 0)

            for i, subgoal in enumerate(solution):
                if i > 0:
                    title = (
                        f'goal_{i}, num={subgoal.child_num}, '
                        f'p={readable_num(subgoal.p)} val={readable_num(subgoal.value)}'
                    )
                    draw_and_log(subgoal.state, title, f'episode_{step}', i)


def solve_board(solver, input_board, use_graph_tracer, recursion_limit=None):
    """
    Constructs model and runs solve. Sets recursion limit for the worker if needed.
    """
    if recursion_limit:
        curr_recursion_limit = sys.getrecursionlimit()

        if curr_recursion_limit < recursion_limit:
            print(f'Rising recursion limit for worker from {curr_recursion_limit} to {recursion_limit}.')
            sys.setrecursionlimit(recursion_limit)

    solver.construct_networks()
    time_s = time.time()
    solution, tree_metrics, root, trajectory_actions, additional_info = solver.solve(input_board, use_graph_tracer)
    time_solving = time.time() - time_s

    return dict(
        solution=solution,
        tree_metrics=tree_metrics,
        root=root,
        trajectory_actions=trajectory_actions,
        time_solving=time_solving,
        input_board=input_board.copy(),
        additional_info=additional_info
    )
