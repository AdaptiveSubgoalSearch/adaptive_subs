import os

import gin
import joblib

from metric_logging import log_text


def get_chunk_filename(file_number):
    return f'chunk_{file_number:08d}'


@gin.configurable
def identity_fn(x):
    return x


@gin.configurable
def arithmetic_sequence_fn(idx, init_number=gin.REQUIRED, difference=gin.REQUIRED):
    return init_number + idx * difference


class LongListDumper:
    def __init__(self, dir_path, compress=9, file_number_fn=identity_fn):
        self._compress = compress

        os.makedirs(dir_path, exist_ok=True)
        self._dir_path = dir_path

        self._file_number_fn = file_number_fn
        self._n_files = 0

    def dump(self, chunk):
        file_number = self._file_number_fn(self._n_files)
        path = os.path.join(self._dir_path, get_chunk_filename(file_number))
        joblib.dump(chunk, path, compress=self._compress)
        self._n_files += 1


class LongListLoader:
    LOG_NAME = 'long_list_loader'

    def __init__(self, dir_path):
        self._dir_path = dir_path
        assert os.path.isdir(self._dir_path)

        self._leftover_elts = []
        self._next_file_number = 0

        available_files = self.count_consecutive_files()
        log_text(
            self.LOG_NAME,
            f'For now there are {available_files} files available.'
        )
        assert available_files > 0, 'Can not find any files to load.'

    def count_consecutive_files(self):
        n_files = 0
        while os.path.isfile(
                os.path.join(self._dir_path, get_chunk_filename(n_files))
        ):
            n_files += 1
        return n_files

    def load(self, n_elts):
        result = self._leftover_elts[:n_elts]
        self._leftover_elts = self._leftover_elts[n_elts:]

        remaining_elts = n_elts - len(result)
        while remaining_elts > 0:
            path = os.path.join(
                self._dir_path,
                get_chunk_filename(self._next_file_number)
            )
            if not os.path.exists(path):
                break
            log_text(self.LOG_NAME, f'Loading file {path}')
            cur_chunk = joblib.load(path)

            result.extend(cur_chunk[:remaining_elts])
            self._leftover_elts = cur_chunk[remaining_elts:]
            remaining_elts = n_elts - len(result)

            self._next_file_number += 1

        return result
