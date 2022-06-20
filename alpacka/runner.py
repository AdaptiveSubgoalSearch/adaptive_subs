# pylint: disable=wrong-import-position
"""Entrypoint of the experiment."""

# Remove the working directory from sys.path - it causes circular imports due to
# module name clashes.
import sys  # isort:skip
sys.path.pop(0)

import argparse
import functools
import itertools
import os
import time

import gin

from alpacka import agents
from alpacka import batch_steppers
from alpacka import envs
from alpacka import metric_logging
from alpacka import networks
from alpacka import trainers
from alpacka.utils import gin as gin_utils
from alpacka.utils import metric as metric_utils
from alpacka.utils import os as os_utils


@gin.configurable
class Runner:
    """Main class running the experiment."""

    def __init__(
        self,
        output_dir,
        env_class=envs.CartPole,
        env_kwargs=None,
        agent_class=agents.RandomAgent,
        network_class=networks.DummyNetwork,
        n_envs=16,
        episode_time_limit=None,
        batch_stepper_class=batch_steppers.LocalBatchStepper,
        trainer_class=trainers.DummyTrainer,
        callback_classes=(),
        n_epochs=None,
        n_precollect_epochs=0,
        metric_smoothing=None,
    ):
        """Initializes the runner.

        Args:
            output_dir (str): Output directory for the experiment.
            env_class (type): Environment class.
            env_kwargs (dict): Keyword arguments to pass to the env class
                when created. It ensures that only the env in the Runner will be
                initialized with them.
            agent_class (type): Agent class.
            network_class (type): Network class.
            n_envs (int): Number of environments to run in parallel.
            episode_time_limit (int or None): Time limit for solving an episode.
                None means no time limit.
            batch_stepper_class (type): BatchStepper class.
            trainer_class (type): Trainer class.
            callback_classes (tuple): Sequence of callback classes to call.
            n_epochs (int or None): Number of epochs to run for, or indefinitely
                if None.
            n_precollect_epochs (int): Number of initial epochs to run without
                training (data precollection).
            metric_smoothing (tuple): Specification of smoothing of metrics.
                (regex, coeff), where of regex is a regex of metrics names which
                should be smoothed. coeff in (0,1) indices the desired smoothing
                parameter. See LoggerWithSmoothing for details.
        """
        self._output_dir = os.path.expanduser(output_dir)
        os.makedirs(self._output_dir, exist_ok=True)

        env_kwargs = env_kwargs or {}
        self._env_fn = functools.partial(env_class, **env_kwargs)

        self._agent_class = agent_class

        self._network_signature = self._infer_network_signature(
            env_class, agent_class
        )
        self._network_fn = functools.partial(
            network_class, network_signature=self._network_signature
        )

        self._n_envs = n_envs
        self._episode_time_limit = episode_time_limit

        self._batch_stepper_class = batch_stepper_class
        self._trainer_class = trainer_class
        self._callback_classes = callback_classes

        self._n_epochs = n_epochs
        self._n_precollect_epochs = n_precollect_epochs
        self._epoch = 0
        self._total_episodes = 0
        self._last_save_time = 0
        self._metric_smoothing = metric_smoothing

        self.reset()

    def reset(self):
        """Reinitializes all components."""
        self._batch_stepper = self._batch_stepper_class(
            env_class=self._env_fn,
            agent_class=self._agent_class,
            network_fn=self._network_fn,
            n_envs=self._n_envs,
            output_dir=self._output_dir,
        )
        self._trainer = self._trainer_class(self._network_signature)
        self._network = self._network_fn()
        self._callbacks = tuple(
            callback_class(runner=self)
            for callback_class in self._callback_classes
        )

    @property
    def _epoch_path(self):
        return os.path.join(self._output_dir, 'epoch')

    @property
    def _network_path(self):
        return os.path.join(self._output_dir, 'network')

    @property
    def _trainer_path(self):
        return os.path.join(self._output_dir, 'trainer')

    @property
    def env_fn(self):
        """Function () -> Env."""
        return self._env_fn

    @property
    def agent_class(self):
        """Agent class."""
        return self._agent_class

    @property
    def network_fn(self):
        """Function () -> Network."""
        return self._network_fn

    @property
    def batch_stepper_class(self):
        """BatchStepper class."""
        return self._batch_stepper_class

    @property
    def output_dir(self):
        """Output directory of the experiment."""
        return self._output_dir

    @staticmethod
    def _infer_network_signature(env_class, agent_class):
        # Initialize an environment and an agent to get a network signature.
        env = env_class()
        agent = agent_class()
        return agent.network_signature(env.observation_space, env.action_space)

    def _save_gin(self):
        config_path = os.path.join(self._output_dir, 'config.gin')
        config_str = gin.operative_config_str()
        with open(config_path, 'w') as f:
            f.write(config_str)

        for (name, value) in gin_utils.extract_bindings(config_str):
            metric_logging.log_property(name, value)

    def run_epoch(self):
        """Runs a single epoch."""
        start_time = time.time()
        episodes = self._batch_stepper.run_episode_batch(
            self._network.params,
            epoch=max(0, self._epoch - self._n_precollect_epochs),
            time_limit=self._episode_time_limit
        )
        episode_metrics = {
            'count': self._total_episodes,
            'time': time.time() - start_time,
        }
        episode_metrics.update(metric_utils.compute_episode_metrics(episodes))
        self._total_episodes += len(episodes)
        metric_logging.log_scalar_metrics(
            'episode', self._epoch, episode_metrics, self._metric_smoothing
        )
        metric_logging.log_scalar_metrics(
            'agent',
            self._epoch,
            self._agent_class.compute_metrics(episodes),
            self._metric_smoothing
        )

        for episode in episodes:
            self._trainer.add_episode(episode)

        if self._epoch >= self._n_precollect_epochs:
            metrics = self._trainer.train_epoch(self._network)
            metric_logging.log_scalar_metrics(
                'train',
                self._epoch,
                metrics,
                self._metric_smoothing
            )

        for callback in self._callbacks:
            callback.on_epoch_end(self._epoch, self._network.params)

        if self._epoch == self._n_precollect_epochs:
            # Save gin operative config into a file. "Operative" means the part
            # that is actually used in the experiment. We need to run an full
            # epoch (data collection + training) first, so gin can figure that
            # out.
            self._save_gin()

        self._epoch += 1

        if time.time() - self._last_save_time > 3600 \
                or self._epoch == self._n_epochs:
            self.save()

    def run(self):
        """Runs the main loop."""
        self.restore()

        if self._n_epochs is None:
            epochs = itertools.repeat(None)  # Infinite stream of Nones.
        else:
            epochs = range(self._epoch, self._n_epochs)

        for _ in epochs:
            self.run_epoch()

    def save(self):
        """Saves the Runner state in output_dir."""
        # Dump the trainer, the network and the epoch number atomically.
        with os_utils.atomic_dump((
            self._epoch_path, self._network_path, self._trainer_path
        )) as (epoch_path, network_path, trainer_path):
            with open(epoch_path, 'w') as f:
                f.write(str(self._epoch))

            self._network.save(network_path)
            self._trainer.save(trainer_path)
            self._last_save_time = time.time()

    def restore(self):
        """Restores the Runner state from output_dir."""
        # Restore from a previous checkpoint, if possible.
        if os.path.exists(self._epoch_path):
            with open(self._epoch_path, 'r') as f:
                self._epoch = int(f.read().strip())

            self._network.restore(self._network_path)
            self._trainer.restore(self._trainer_path)

    def close(self):
        """Cleans up the resources."""
        self._batch_stepper.close()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir', required=True,
        help='Output directory.')
    parser.add_argument(
        '--config_file', action='append',
        help='Gin config files.'
    )
    parser.add_argument(
        '--config', action='append',
        help='Gin config overrides.'
    )
    parser.add_argument(
        '--mrunner', action='store_true',
        help='Add mrunner spec to gin-config overrides and Neptune to loggers.'
        '\nNOTE: It assumes that the last config override (--config argument) '
        'is a path to a pickled experiment config created by the mrunner CLI or'
        'a mrunner specification file.'
    )
    parser.add_argument(
        '--tensorboard', action='store_true',
        help='Enable TensorBoard logging: logdir=<output_dir>/tb_%m-%dT%H%M%S.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    gin_bindings = args.config

    if args.mrunner:
        from alpacka.utils import mrunner_client  # Lazy import
        spec_path = gin_bindings.pop()

        specification, overrides = mrunner_client.get_configuration(spec_path)
        gin_bindings = overrides + gin_bindings

        try:
            neptune_logger = mrunner_client.configure_neptune(specification)
            metric_logging.register_logger(neptune_logger)

        except mrunner_client.NeptuneAPITokenException:
            print('HINT: To run with Neptune logging please set your '
                  'NEPTUNE_API_TOKEN environment variable')

    if args.tensorboard:
        from alpacka.utils import tensorboard  # Lazy import

        tensorboard_logger = tensorboard.TensorBoardLogger(args.output_dir)
        metric_logging.register_logger(tensorboard_logger)

    gin.parse_config_files_and_bindings(args.config_file, gin_bindings)
    runner = Runner(args.output_dir)
    runner.run()
    runner.close()
