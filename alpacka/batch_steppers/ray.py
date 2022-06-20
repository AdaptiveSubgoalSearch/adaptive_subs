"""Ray-distributed environment stepper."""

import random
import typing

import gin

from alpacka import data
from alpacka.batch_steppers import core
from alpacka.batch_steppers import worker_utils


def _ray():
    """Imports ray lazily to avoid import order issues."""
    # Ray must be imported before pickle5, which is imported by cloudpickle,
    # which is used in ProcessBatchStepper:
    # https://github.com/ray-project/ray/blob/622eee45ddee98b81dc6d9fb2f8af0436a04704f/python/ray/__init__.py#L12
    import ray  # pylint: disable=import-outside-toplevel
    return ray


class RayObject(typing.NamedTuple):
    """Keeps value and id of an object in the Ray Object Store."""
    id: typing.Any
    value: typing.Any

    @classmethod
    def from_value(cls, value, weakref=False):
        return cls(_ray().put(value, weakref=weakref), value)


class RayBatchStepper(core.BatchStepper):
    """Batch stepper running remotely using Ray.

    Runs predictions and steps environments for all Agents separately in their
    own workers.

    It's highly recommended to pass params to run_episode_batch as a numpy array
    or a collection of numpy arrays. Then each worker can retrieve params with
    zero-copy operation on each node.
    """

    def __init__(
        self,
        env_class,
        agent_class,
        network_fn,
        n_envs,
        output_dir,
        compress_episodes=True,
    ):
        super().__init__(env_class, agent_class, network_fn, n_envs, output_dir)

        config = worker_utils.get_config(env_class, agent_class, network_fn)
        ray_worker_cls = _ray().remote(worker_utils.Worker)

        if not _ray().is_initialized():
            kwargs = {
                # Size of the Plasma object store, hardcoded to 1GB for now.
                'object_store_memory': int(1e9),
                # Override the Ray log dir to a random one - otherwise it uses
                # /tmp/ray, which occasionally causes permission problems.
                # Interestingly, logging to output_dir causes segfaults (???).
                'temp_dir': '/tmp/ray{}'.format(random.randrange(1e3)),
            }
            _ray().init(**kwargs)

        self.workers = [
            ray_worker_cls.remote(  # pylint: disable=no-member
                env_class=env_class,
                agent_class=agent_class,
                network_fn=network_fn,
                config=config,
                scope=gin.current_scope(),
                init_hooks=worker_utils.init_hooks,
                compress_episodes=compress_episodes,
            )
            for _ in range(n_envs)
        ]

        self._params = RayObject(None, None)
        self._solve_kwargs_per_worker = [
            RayObject(None, None) for _ in range(self.n_envs)
        ]
        self._compress_episodes = compress_episodes

    def _run_episode_batch(self, params, solve_kwargs_per_agent):
        # Optimization, don't send the same parameters again.
        if self._params.value is None or not data.nested_array_equal(
                params, self._params.value
        ):
            self._params = RayObject.from_value(params)
        
        self._solve_kwargs_per_worker = [
            RayObject.from_value(solve_kwargs)
            for solve_kwargs in solve_kwargs_per_agent
        ]

        episodes = _ray().get([
            w.run.remote(self._params.id, solve_kwargs.id)
            for w, solve_kwargs in
            zip(self.workers, self._solve_kwargs_per_worker)]
        )
        if self._compress_episodes:
            episodes = [
                worker_utils.decompress_episode(episode)
                for episode in episodes
            ]
        return episodes

    def close(self):
        """Closes the workers."""
        _ray().get([w.close.remote() for w in self.workers])
