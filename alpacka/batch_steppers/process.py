"""Process-distributed environment stepper."""

import collections
import functools
import multiprocessing as _mp
import sys

import gin
from tblib import pickling_support

from alpacka.batch_steppers import core
from alpacka.batch_steppers import worker_utils


# Spawn new processes instead of forking - the latter causes problems with TF.
mp = _mp.get_context(method='spawn')


pickling_support.install()


_ExcInfo = collections.namedtuple('_ExcInfo', ('type', 'value', 'traceback'))


def _cloudpickle():
    """Imports cloudpickle lazily to avoid import order issues."""
    # Ray must be imported before pickle5, which is imported by cloudpickle:
    # https://github.com/ray-project/ray/blob/622eee45ddee98b81dc6d9fb2f8af0436a04704f/python/ray/__init__.py#L12
    import cloudpickle  # pylint: disable=import-outside-toplevel
    return cloudpickle


class ProcessBatchStepper(core.BatchStepper):
    """BatchStepper running in multiple processes.

    Runs predictions and steps environments for all Agents separately in their
    own workers.
    """

    def __init__(
        self,
        env_class,
        agent_class,
        network_fn,
        n_envs,
        output_dir,
        process_class=mp.Process,
        serialize_worker_fn=True,
    ):
        """Initializes ProcessBatchStepper.

        Args:
            env_class (type): Environment class.
            agent_class (type): Agent class.
            network_fn (callable): Function () -> Network.
            n_envs (int): Number of parallel environments to run.
            output_dir (str or None): Experiment output dir if the BatchStepper
                is initialized from Runner, None otherwise.
            process_class (type): Class with the same interface as
                multiprocessing.Process, used to spawn workers. Settable for
                testing purposes.
            serialize_worker_fn (bool): Whether the worker factory function
                should be serialized to a bytestring before sending to the
                processes. This is used to workaround the incompatibility
                between gin and the standard pickle module, used internally
                by multiprocessing. Settable for testing purposes.
        """
        super().__init__(env_class, agent_class, network_fn, n_envs, output_dir)

        config = worker_utils.get_config(env_class, agent_class, network_fn)

        def start_worker():
            worker_fn = functools.partial(
                worker_utils.Worker,
                env_class=env_class,
                agent_class=agent_class,
                network_fn=network_fn,
                config=config,
                scope=gin.current_scope(),
                # Not supported yet because of Neptune issues.
                init_hooks=[],
                compress_episodes=False,
            )
            # Serialize the worker_fn using cloudpickle - standard pickle
            # used by multiprocessing breaks gin configurables.
            if serialize_worker_fn:
                worker_fn = _cloudpickle().dumps(worker_fn)
            queue_in = mp.Queue()
            queue_out = mp.Queue()
            process = process_class(
                target=_target,
                args=(worker_fn, queue_in, queue_out, serialize_worker_fn),
            )
            process.start()
            return (queue_in, queue_out)

        self._queues = [start_worker() for _ in range(n_envs)]
        # Intercept any init errors.
        for (_, queue_out) in self._queues:
            _receive(queue_out)

    def _run_episode_batch(self, params, solve_kwargs_per_agent):
        for ((queue_in, _), solve_kwargs) in zip(
            self._queues, solve_kwargs_per_agent
        ):
            queue_in.put((params, solve_kwargs))

        return [_receive(queue_out) for (_, queue_out) in self._queues]

    def close(self):
        # Send a shutdown message to all processes.
        for (queue_in, _) in self._queues:
            queue_in.put(None)


def _target(worker_fn, queue_in, queue_out, serialized):
    try:
        if serialized:
            worker_fn = _cloudpickle().loads(worker_fn)
        worker = worker_fn()
        queue_out.put(None)  # Signalize that everything's ok.
        while True:
            msg = queue_in.get()
            if msg is None:  # None means shutdown.
                worker.close()
                break
            (params, solve_kwargs) = msg
            episode = worker.run(params, solve_kwargs)
            queue_out.put(episode)
    except Exception:  # pylint: disable=broad-except
        queue_out.put(_ExcInfo(*sys.exc_info()))


def _receive(queue):
    msg = queue.get()
    if isinstance(msg, _ExcInfo):
        # Oops, exception in a worker. Propagate.
        exc = msg.value
        raise exc.with_traceback(msg.traceback)
    return msg
