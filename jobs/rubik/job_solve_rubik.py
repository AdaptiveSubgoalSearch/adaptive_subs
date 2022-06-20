import pickle
import sys
import time
from copy import deepcopy

import cloudpickle

from envs import Sokoban

from joblib import Parallel, delayed
from jobs.core import Job
from metric_logging import log_scalar, log_scalar_metrics, MetricsAccumulator, log_text

# from visualization.seq_parse import logic_statement_to_seq_string, entity_to_seq_string
from supervised.rubik.rubik_solver_utils import generate_problems_rubik, cube_to_string


def solve_problem(solver, input_state, recursion_limit):
    if recursion_limit:
        curr_recursion_limit = sys.getrecursionlimit()
        if curr_recursion_limit < recursion_limit:
            print(f'Rising recursion limit for worker from {curr_recursion_limit} to {recursion_limit}.')
            sys.setrecursionlimit(recursion_limit)

    time_s = time.time()
    solution, tree_metrics, root, trajectory_actions, additional_info = solver.solve(input_state)
    time_solving = time.time() - time_s

    return dict(
        solution=solution,
        tree_metrics=tree_metrics,
        root=root,
        trajectory_actions=trajectory_actions,
        time_solving=time_solving,
        input_problem=deepcopy(input_state),
        additional_info=additional_info
    )


class JobSolveRubik(Job):
    def __init__(self,
                 solver_class,
                 n_jobs,
                 n_parallel_workers,
                 batch_size,
                 budget_checkpoints=None,
                 real_budget_checkpoints=None,
                 log_solutions_limit=100,
                 job_range=None,
                 collect_solutions=None,
                 recursion_limit=None,
                 ):

        self.solver_class = solver_class
        self.n_jobs = n_jobs
        self.n_parallel_workers = n_parallel_workers
        self.batch_size = batch_size
        self.budget_checkpoints = budget_checkpoints
        self.real_budget_checkpoints = real_budget_checkpoints
        self.log_solutions_limit = log_solutions_limit
        self.job_range = job_range
        self.collect_solution = collect_solutions
        self.recursion_limit = recursion_limit

        self.solved_stats = MetricsAccumulator()
        self.experiment_stats = MetricsAccumulator()
        self.verificator_stats = MetricsAccumulator()
        self.calls_stats = MetricsAccumulator()
        self.distance_stats = MetricsAccumulator()

        self.verifications = dict()
        self.subgoals_tested = 0
        self.subgoals_skipped = 0

        self.logged_solutions = 0

        if self.collect_solution is not None:
            self.collection = {}

    def execute(self):
        import numpy as np
        np.random.seed(234856566)  # NOTE temporary solution
        proofs_to_solve = generate_problems_rubik(self.n_jobs)
        problems_to_solve = [proof[0] for proof in proofs_to_solve]

        solver = self.solver_class()
        solver.construct_networks()
        jobs_done = 0
        jobs_to_do = self.n_jobs
        batch_num = 0
        all_batches = self.n_jobs // self.batch_size

        total_time_start = time.time()
        while jobs_to_do > 0:
            jobs_in_batch = min(jobs_to_do, self.batch_size)
            problems_to_solve_in_batch = problems_to_solve[jobs_done:jobs_done+jobs_in_batch]
            print('============================ Batch {:>4}  out  of  {:>4} ============================'.
                  format(batch_num+1, all_batches))
            results = Parallel(n_jobs=self.n_parallel_workers, verbose=100)(
                delayed(solve_problem)(solver, input_problem, self.recursion_limit)
                for input_problem in problems_to_solve_in_batch
            )
            # results = [
            #     solve_problem(solver, input_problem) for input_problem in problems_to_solve_in_batch
            # ]
            print('===================================================================================')
            self.log_results(results, jobs_done)
            jobs_done += jobs_in_batch
            jobs_to_do -= jobs_in_batch
            batch_num += 1

        for metric, value in self.solved_stats.return_scalars().items():
            log_text('summary', f'{metric},  {value}')
        log_text('summary', f'Finished time , {time.time() - total_time_start}')

    def log_results(self, results, step):
        n_logs = len(results)

        for log_num, result in enumerate(results):
            log_scalar_metrics('tree', step+log_num, result['tree_metrics'])
            if self.logged_solutions < self.log_solutions_limit:
                self.log_solution(result['solution'], result['trajectory_actions'], result['input_problem'], step+log_num)
            solved = result['solution'] is not None
            self.experiment_stats.log_metric_to_accumulate('tested', 1)
            log_scalar_metrics('problems', step+log_num, self.experiment_stats.return_scalars())
            log_scalar('time_solving', step + log_num, result['time_solving'])

            if solved:
                self.solved_stats.log_metric_to_average('rate/all', 1)
                self.solved_stats.log_metric_to_accumulate('problems', 1)
                log_scalar('solution', step + log_num, 1)
                log_scalar('solution/length', step + log_num, len(result['trajectory_actions']))
                # assert False
                trajectory_actions = [str(action) for action in result['trajectory_actions']]
                trajectory = ', '.join(trajectory_actions)
                log_text('trajectory_actions', f'{step + log_num}: {trajectory}', False)
                log_scalar('solution/n_subgoals', step + log_num, len(result['solution']))

                log_scalar('verificator min prob on final trajectory', step + log_num, result['additional_info']['verificator_min_prob'])

                # subgoal distances
                path = list(reversed(result['additional_info']['subgoal_distances']))
                if len(path) >= 1:
                    for id in [0, 1, -1, -2]:
                        log_scalar(f'solution/distances/step nb. {id}', step + log_num, path[id])
                        self.distance_stats.log_metric_to_average(f'avg step nb. {id}', path[id])
            else:
                self.solved_stats.log_metric_to_average('rate/all', 0)
                self.solved_stats.log_metric_to_accumulate('problems', 0)
                log_scalar('solution', step+log_num, 0)
                log_scalar('solution/length', step + log_num, -1)
                log_text('trajectory_actions', f'{step + log_num}: unsolved', False)
                log_scalar('solution/n_subgoals', step + log_num, -1)

            log_scalar('verificator failed', step + log_num, result['additional_info']['verificator_failed'])
            self.subgoals_tested += result['additional_info']['subgoals_tested']
            self.subgoals_skipped += result['additional_info']['subgoals_skipped']

            joint_calls = 0
            for key in result['additional_info']:
                if 'calls/' in key:
                    self.calls_stats.log_metric_to_accumulate(key + '_sum', result['additional_info'][key])
                    log_scalar(key, step + log_num, result['additional_info'][key])
                    joint_calls += result['additional_info'][key]
            self.calls_stats.log_metric_to_accumulate('calls/joint_sum', joint_calls)
            log_scalar('calls/joint', step + log_num, joint_calls)

            for i, verification in enumerate(result['additional_info']['verifications']):
                if i in self.verifications.keys():
                    self.verifications[i] = self.verifications[i] + verification
                else:
                    self.verifications[i] = verification

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

            if self.real_budget_checkpoints is not None:
                for budget in self.real_budget_checkpoints:
                    if result['tree_metrics']['real_cost_final'] <= budget and solved:
                        self.solved_stats.log_metric_to_average(f'rate/{budget}_evaluations', 1)
                    else:
                        self.solved_stats.log_metric_to_average(f'rate/{budget}_evaluations', 0)

        for i, verification in self.verifications.items():
            log_text(f'verifications of builder {i}', str(verification), False)
            log_text(f'(normalized) verifications of builder {i}',
                     str([verification[0] / sum(verification[0]), verification[1] / sum(verification[1])]), False)

        log_scalar('subgoals tested', step+n_logs, self.subgoals_tested)
        log_scalar('subgoals skipped', step+n_logs, self.subgoals_skipped)
        log_scalar_metrics('solved', step+n_logs, self.solved_stats.return_scalars())
        log_scalar_metrics('calls', step+n_logs, self.calls_stats.return_scalars())
        log_scalar_metrics('solution/distances', step+n_logs, self.distance_stats.return_scalars())

    def log_solution(self, solution, trajectory_actions, input_problem, step):
        if solution is not None:
            solution_str = f'Problem {step} : {solution[0].hash} \n'
            for subgoal_num, node in enumerate(solution[1:]):
                solution_str += f'subgoal {subgoal_num} : {node.hash} \n'
            solution_str += '\n \n'
            solution_str += 'Actions: \n'
            for action_num, action in enumerate(trajectory_actions):
                # solution_str += f'action {action_num}: ({action[0]}, {[entity_to_seq_string(ent) for ent in action[1:]]} ) \n'
                solution_str += f'action {action_num}: ({action}, () ) \n'

        else:
            solution_str = f'Unsolved problem {step} : {input_problem} \n \n'

        log_text('solution', solution_str, True)

        if self.collect_solution is not None:
            if solution is not None:
                new_path = [(node.hash, -node.value) for node in solution[:-1]]
                new_path.append((solution[-1].hash, 0))
            else:
                new_path = None
            self.collection[len(self.collection)] = new_path
            print(new_path)
            pickle.dump(self.collection, open(self.collect_solution, "wb"))
