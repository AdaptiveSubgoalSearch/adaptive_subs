"""Runner callbacks."""


import functools

import gin

from alpacka import metric_logging
from alpacka.utils import metric as metric_utils


class RunnerCallback:
    """Base class for Runner callbacks."""

    def __init__(self, runner):
        self._runner = runner

    def on_epoch_end(self, epoch, network_params):
        """Called at the end of each epoch."""


@gin.configurable
class EvaluationCallback(RunnerCallback):
    """Callback for running Agent evaluation.

    Can override agent's init kwargs, for example to turn off exploration.
    """

    def __init__(
        self,
        n_envs=1,
        episode_time_limit=None,
        eval_period=10,
        agent_kwargs=None,
        env_kwargs=None,
        metric_smoothing=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._episode_time_limit = episode_time_limit
        self._eval_period = eval_period
        agent_kwargs = agent_kwargs or {}
        env_kwargs = env_kwargs or {}
        self._metric_smoothing = metric_smoothing
        self._batch_stepper = self._runner.batch_stepper_class(
            env_class=functools.partial(self._runner.env_fn, **env_kwargs),
            agent_class=functools.partial(
                self._runner.agent_class, **agent_kwargs
            ),
            network_fn=self._runner.network_fn,
            n_envs=n_envs,
            output_dir=self._runner.output_dir,
        )

        # We add the gin scope to the metric names, so we can have multiple eval
        # callbacks reporting separate metrics.
        if gin.current_scope():
            self._scope = '/' + gin.current_scope_str()
        else:
            self._scope = ''

    def on_epoch_end(self, epoch, network_params):
        if epoch % self._eval_period != 0:
            return

        episodes = self._batch_stepper.run_episode_batch(
            network_params,
            epoch=epoch,
            time_limit=self._episode_time_limit
        )

        metric_logging.log_scalar_metrics(
            'eval_agent' + self._scope,
            epoch,
            self._runner.agent_class.compute_metrics(episodes)
        )

        metric_logging.log_scalar_metrics(
            'eval_episode' + self._scope,
            epoch,
            metric_utils.compute_episode_metrics(episodes),
            self._metric_smoothing,
        )


@gin.configurable
class ReconfigurationCallback(RunnerCallback):
    """Callback for changing the configuration during training."""

    def __init__(self, runner, epoch_to_scope=None):
        """Initializes ReconfigCallback.

        Args:
            epoch_to_scope: Dict epoch -> gin scope. The callback will
                reconfigure Runner using config in the specified scopes at the
                specified epochs.
            runner: The Runner instance.
        """
        super().__init__(runner)
        self._epoch_to_scope = epoch_to_scope or {}

    def on_epoch_end(self, epoch, network_params):
        del network_params

        # Check the next epoch so the new config is introduced at the beginning
        # of that epoch.
        scope = self._epoch_to_scope.get(epoch + 1)
        if scope is not None:
            with gin.config_scope(scope):
                # Save the training state, so we can restore it after
                # reconfiguration.
                self._runner.save()
                # Reset the Runner using the new configuration.
                self._runner.reset()
                # Restore the training state.
                self._runner.restore()
