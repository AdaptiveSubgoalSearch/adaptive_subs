import pickle
import time
from copy import deepcopy

import cloudpickle

from envs import Sokoban

from joblib import Parallel, delayed
from jobs.core import Job
from metric_logging import log_scalar, log_scalar_metrics, MetricsAccumulator, log_text
from supervised.int.gen_subgoal_data import generate_problems

from utils.general_utils import readable_num
from utils.utils_sokoban import draw_and_log
from visualization.seq_parse import logic_statement_to_seq_string, entity_to_seq_string


def solve_problem(vanilla_policy, input_state):
    time_s = time.time()
    solved, additional_info = vanilla_policy.solve(input_state)
    time_solving = time.time() - time_s
    return dict(
        solved=solved,
        time_solving=time_solving,
        input_problem=deepcopy(input_state),
        additional_info=additional_info
    )


class JobVanillaSolveINT(Job):
    def __init__(self,
                 n_jobs,
                 vanilla_policy_class=None,
                 budget_checkpoints=None,
                 log_solutions_limit=100,
                 n_parallel_workers=1,
                 batch_size=1,
                 ):

        self.vanilla_policy = vanilla_policy_class()
        self.n_jobs = n_jobs
        self.budget_checkpoints = budget_checkpoints
        self.log_solutions_limit = log_solutions_limit
        self.n_parallel_workers = n_parallel_workers
        self.batch_size = batch_size
        self.budget_checkpoints = budget_checkpoints

        self.solved_stats = MetricsAccumulator()
        self.experiment_stats = MetricsAccumulator()


        self.collection = {}


    def execute(self):

        self.vanilla_policy.construct_networks()
        proofs_to_solve = generate_problems(self.n_jobs)
        jobs_done = 0

        total_time_start = time.time()
        # for job_num in range(self.n_jobs):
        #     print(f'============================ Problem {job_num} ============================')
        #     results = solve_problem(self.vanilla_policy, proofs_to_solve[job_num][0])
        #     print('===================================================================================')
        #     self.log_results(results, jobs_done)
        #     jobs_done += 1

        jobs_done = 0
        jobs_to_do = self.n_jobs
        batch_num = 0

        while jobs_to_do > 0:
            jobs_in_batch = min(jobs_to_do, self.batch_size)
            boards_to_solve_in_batch = proofs_to_solve[jobs_done:jobs_done + jobs_in_batch]

            results = Parallel(n_jobs=self.n_parallel_workers, verbose=100)(
                delayed(solve_problem)(self.vanilla_policy, input_problem[0]) for input_problem in boards_to_solve_in_batch
            )

            self.log_results(results, jobs_done)

            jobs_done += jobs_in_batch
            jobs_to_do -= jobs_in_batch
            batch_num += 1

        for metric, value in self.solved_stats.return_scalars().items():
            log_text('summary', f'{metric},  {value}')
        log_text('summary', f'Finished time , {time.time() - total_time_start}')

    def log_results(self, results, step):
        log_num = 0
        for result in results:
            if result['solved']:
                self.solved_stats.log_metric_to_average('rate', 1)
                self.solved_stats.log_metric_to_accumulate('problems', 1)
                log_scalar('solution', step + log_num, 1)

            else:
                self.solved_stats.log_metric_to_average('rate', 0)
                self.solved_stats.log_metric_to_accumulate('problems', 0)
                log_scalar('solution', step+log_num, 0)

            log_scalar('tree/nodes', step + log_num, result['additional_info']['num_steps'])

            solved = result['solved']
            if self.budget_checkpoints is not None:
                for budget in self.budget_checkpoints:
                    if solved and result['additional_info']['num_steps'] <= budget:
                        self.solved_stats.log_metric_to_average(f'rate/{budget}_evaluations', 1)
                    else:
                        self.solved_stats.log_metric_to_average(f'rate/{budget}_evaluations', 0)

            log_num += 1

        log_scalar_metrics('solved', step+log_num, self.solved_stats.return_scalars())




