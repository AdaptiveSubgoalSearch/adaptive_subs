import random
import time

import numpy as np

from jobs import core
from metric_logging import log_scalar
from supervised.int.gen_subgoal_data import generate_problems
from utils import storage


class JobIntGenerateProblems(core.Job):
    def __init__(
        self, dir_path, n_files, n_problems_per_file=1000,
        file_number_fn=storage.identity_fn,
        compress=9, seed=35,
    ):
        self._dir_path = dir_path
        self._n_files = n_files
        self._n_problems_per_file = n_problems_per_file
        self._file_number_fn = file_number_fn
        self._compress = compress
        self._seed = seed

    def execute(self):
        random.seed(self._seed)
        np.random.seed(self._seed)

        dumper = storage.LongListDumper(
            self._dir_path, compress=self._compress,
            file_number_fn=self._file_number_fn
        )
        for i in range(self._n_files):
            print(f'{i}/{self._n_files}')

            t_gen = time.time()
            # Use gin to specify desired parameters for this function.
            problems = generate_problems(self._n_problems_per_file)
            log_scalar('time_generate', i, time.time() - t_gen)

            t_dump = time.time()
            dumper.dump(problems)
            log_scalar('time_dump', i, time.time() - t_dump)
