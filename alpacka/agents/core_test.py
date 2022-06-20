"""Tests for alpacka.agents.core."""

import collections

import gym
import numpy as np
import pytest

from alpacka import agents
from alpacka import testing
from alpacka import utils


mock_env = testing.mock_env_fixture


@pytest.mark.parametrize('with_critic', [True, False])
@pytest.mark.parametrize('agent_class',
                         [agents.SoftmaxAgent,
                          agents.EpsilonGreedyAgent])
def test_agents_network_signature(agent_class, with_critic):
    # Set up
    obs_space = gym.spaces.Box(low=0, high=255, shape=(7, 7), dtype=np.uint8)
    act_space = gym.spaces.Discrete(n=7)

    # Run
    agent = agent_class(with_critic=with_critic)
    signature = agent.network_signature(obs_space, act_space)

    # Test
    assert signature.input.shape == obs_space.shape
    assert signature.input.dtype == obs_space.dtype
    if with_critic:
        assert signature.output[0].shape == (1, )
        assert signature.output[0].dtype == np.float32
        assert signature.output[1].shape == (act_space.n, )
        assert signature.output[1].dtype == np.float32
    else:
        assert signature.output.shape == (act_space.n, )
        assert signature.output.dtype == np.float32


@pytest.mark.parametrize('agent_class,attr_name',
                         [(agents.SoftmaxAgent, 'distribution.temperature'),
                          (agents.EpsilonGreedyAgent, 'distribution.epsilon')])
def test_agents_linear_annealing_exploration_parameter(
        agent_class, attr_name, mock_env):
    # Set up
    max_value = 1
    min_value = 0
    param_values = np.arange(max_value, min_value, -1)
    n_epochs = len(param_values)

    agent = agent_class(linear_annealing_kwargs={
        'max_value': max_value,
        'min_value': min_value,
        'n_epochs': n_epochs,
    })

    # Run & Test
    for epoch, x_value in enumerate(param_values):
        testing.run_with_constant_network_prediction(
            agent.solve(mock_env, epoch=epoch),
            logits=np.array([[3, 2, 1]])
        )
        assert utils.recursive_getattr(agent, attr_name) == x_value


def test_softmax_agent_action_counts_for_different_temperature():
    # Set up
    low_temp_agent = agents.SoftmaxAgent(temperature=.5)
    high_temp_agent = agents.SoftmaxAgent(temperature=2.)
    low_temp_action_count = collections.defaultdict(int)
    high_temp_action_count = collections.defaultdict(int)
    logits = ((2, 1, 1, 1, 2), )  # Batch of size 1.

    # Run
    for agent, action_count in [
        (low_temp_agent, low_temp_action_count),
        (high_temp_agent, high_temp_action_count),
    ]:
        for _ in range(1000):
            action, _ = testing.run_with_constant_network_prediction(
                agent.act(np.zeros((7, 7))),
                logits
            )
            action_count[action] += 1

    # Test
    assert low_temp_action_count[0] > high_temp_action_count[0]
    assert low_temp_action_count[1] < high_temp_action_count[1]
    assert low_temp_action_count[2] < high_temp_action_count[2]
    assert low_temp_action_count[3] < high_temp_action_count[3]
    assert low_temp_action_count[4] > high_temp_action_count[4]


def test_egreedy_agent_action_counts_for_different_epsilon():
    # Set up
    low_eps_agent = agents.EpsilonGreedyAgent(epsilon=.05)
    high_eps_agent = agents.EpsilonGreedyAgent(epsilon=.5)
    low_eps_action_count = collections.defaultdict(int)
    high_eps_action_count = collections.defaultdict(int)
    logits = ((5, 4, 3, 2, 1), )  # Batch of size 1.

    # Run
    for agent, action_count in [
        (low_eps_agent, low_eps_action_count),
        (high_eps_agent, high_eps_action_count),
    ]:
        for _ in range(1000):
            action, _ = testing.run_with_constant_network_prediction(
                agent.act(np.zeros((7, 7))),
                logits
            )
            action_count[action] += 1

    # Test
    assert low_eps_action_count[0] > high_eps_action_count[0]
    assert low_eps_action_count[1] < high_eps_action_count[1]
    assert low_eps_action_count[2] < high_eps_action_count[2]
    assert low_eps_action_count[3] < high_eps_action_count[3]
    assert low_eps_action_count[4] < high_eps_action_count[4]
