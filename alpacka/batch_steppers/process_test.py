"""Tests for alpacka.batch_steppers.process."""

import functools
import threading

import pytest

from alpacka import batch_steppers
from alpacka import envs
from alpacka import networks


class _TestException(Exception):
    pass


# 3 sec timeout to detect hangups.
@pytest.mark.timeout(timeout=3, method='signal')
def test_propagates_errors():
    class TestAgent:
        def solve(self, env, **_):
            del env
            raise _TestException

    bs = batch_steppers.ProcessBatchStepper(
        env_class=envs.CartPole,
        agent_class=TestAgent,
        network_fn=functools.partial(
            networks.DummyNetwork, network_signature=None
        ),
        n_envs=1,
        output_dir=None,
        process_class=threading.Thread,
    )
    try:
        with pytest.raises(_TestException):
            bs.run_episode_batch(params=None)
    finally:
        bs.close()
