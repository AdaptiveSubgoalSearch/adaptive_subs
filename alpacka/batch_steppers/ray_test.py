"""Tests for alpacka.batch_steppers.ray."""

import functools
import platform
from unittest import mock

import gin
import pytest
import ray

from alpacka import agents
from alpacka import batch_steppers
from alpacka import envs
from alpacka import networks
from alpacka.batch_steppers import worker_utils


class _TestWorker(worker_utils.Worker):

    def get_state(self):
        return self.env, self.agent, self.network


@gin.configurable
class _TestAgent(agents.RandomAgent):

    def __init__(self, test_attr=None):
        super().__init__()
        self.test_attr = test_attr


@pytest.mark.parametrize('compress_episodes', [False, True])
def test_integration_with_cartpole(compress_episodes):
    n_envs = 3

    bs = batch_steppers.RayBatchStepper(
        env_class=envs.CartPole,
        agent_class=agents.RandomAgent,
        network_fn=functools.partial(
            networks.DummyNetwork, network_signature=None
        ),
        n_envs=n_envs,
        output_dir=None,
        compress_episodes=compress_episodes,
    )
    episodes = bs.run_episode_batch(params=None, time_limit=10)

    assert len(episodes) == n_envs
    for episode in episodes:
        assert hasattr(episode, 'transition_batch')


@mock.patch('alpacka.batch_steppers.worker_utils.Worker', _TestWorker)
@pytest.mark.skipif(platform.system() == 'Darwin',
                    reason='Ray does not work on Mac, see awarelab/alpacka#27')
def test_ray_batch_stepper_worker_members_initialization_with_gin_config():
    # Set up
    test_attr = 7
    env_class = envs.CartPole
    agent_class = _TestAgent
    network_class = networks.DummyNetwork
    network_fn = functools.partial(
        networks.DummyNetwork, network_signature=None
    )
    n_envs = 3

    gin.bind_parameter('_TestAgent.test_attr', test_attr)

    agent = agent_class()
    assert agent.test_attr == test_attr

    # Run
    bs = batch_steppers.RayBatchStepper(
        env_class=env_class,
        agent_class=agent_class,
        network_fn=network_fn,
        n_envs=n_envs,
        output_dir=None,
    )
    bs.run_episode_batch(params=None, time_limit=10)

    # Test
    assert agent.test_attr == test_attr
    assert len(bs.workers) == n_envs
    for worker in bs.workers:
        env, agent, network = ray.get(worker.get_state.remote())
        assert isinstance(env, env_class)
        assert isinstance(agent, agent_class)
        assert isinstance(network, network_class)
        assert agent.test_attr == test_attr

    # Clean up.
    bs.close()
