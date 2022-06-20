import os
import logging
import platform
import sys
sys.path.append("./third_party/INT")
sys.path.insert(0, "/additional_packages")
# To import alpacka stuff do not use `import third_party.alpacka...`, but simply
# `import alpacka...`. Otherwise __init__.py files will be called twice, which
# will cause gin errors.
sys.path.append("./third_party")

# Copied from silence_tensorflow (https://pypi.org/project/silence-tensorflow/) to suppress warnings:
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

import torch

import argparse
from third_party import dask


import gin
# This makes gin configurable classes picklable
gin.config._OPERATIVE_CONFIG_LOCK = dask.SerializableLock()

import envs
import goal_builders
import goal_generating_networks
import goal_verificators
import graph_tracer
import jobs
import policies
import solvers
import supervised
import value_estimators

import metric_logging


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file', action='append', default=[],
        help='Gin config files.'
    )
    parser.add_argument(
        '--config', action='append', default=[],
        help='Gin config overrides.'
    )
    parser.add_argument(
        '--mrunner', action='store_true',
        help='Add mrunner spec to gin-config overrides and Neptune to loggers.'
        '\nNOTE: It assumes that the last config override (--config argument) '
        'is a path to a pickled experiment config created by the mrunner CLI or'
        'a mrunner specification file.'
    )
    return parser.parse_args()

@gin.configurable()
def run(job_class):
    metric_logging.log_text('host_name', platform.node())
    metric_logging.log_text('n_gpus', str(torch.cuda.device_count()))

    job = job_class()
    return job.execute()

if __name__ == '__main__':
    args = _parse_args()

    gin_bindings = args.config
    if args.mrunner:
        from mrunner_utils import mrunner_client
        spec_path = gin_bindings.pop()
        specification, overrides = mrunner_client.get_configuration(spec_path)
        gin_bindings.extend(overrides)

        if 'use_neptune' in specification['parameters']:
            if specification['parameters']['use_neptune']:
                try:
                    neptune_logger = mrunner_client.configure_neptune(specification)
                    metric_logging.register_logger(neptune_logger)

                except mrunner_client.NeptuneAPITokenException:
                    print('HINT: To run with Neptune logging please set your '
                          'NEPTUNE_API_TOKEN environment variable')

    gin.parse_config_files_and_bindings(args.config_file, gin_bindings)
    run()
