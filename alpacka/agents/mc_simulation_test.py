"""Tests for alpacka.agents.mc_simulation."""

import math
from unittest import mock

import numpy as np
import pytest

from alpacka import agents
from alpacka import batch_steppers
from alpacka import envs
from alpacka import testing
from alpacka.agents import mc_simulation


@pytest.mark.parametrize('truncated,x_return',
                         [(False, 2 - 1),
                          (True, 2 - 1 + 7)])
def test_bootstrap_return_with_value_estimator(truncated, x_return):
    # Set up
    episode = testing.construct_episodes(
        actions=[
            [0, 1, 2, 3],
        ],
        rewards=[
            [0, 2, 0, -1]
        ],
        truncated=truncated
    )[0]
    logits = (np.array([[7]]), None)

    # Run
    bootstrap_return = testing.run_with_constant_network_prediction(
        mc_simulation.bootstrap_return_with_value([episode]),
        logits=logits
    )[0]

    # Test
    assert bootstrap_return == x_return


def test_integration_with_cartpole():
    # Set up
    env = envs.CartPole()
    agent = agents.ShootingAgent(n_rollouts=1)

    # Run
    episode = testing.run_with_dummy_network_response(agent.solve(env))

    # Test
    assert episode.transition_batch.observation.shape[0]  # pylint: disable=no-member


def test_act_doesnt_change_env_state():
    # Set up
    env = envs.CartPole()
    agent = agents.ShootingAgent(n_rollouts=10)
    observation = env.reset()
    testing.run_with_dummy_network_response(agent.reset(env, observation))

    # Run
    state_before = env.clone_state()
    testing.run_without_suspensions(agent.act(observation))
    state_after = env.clone_state()

    # Test
    np.testing.assert_equal(state_before, state_after)


mock_env = testing.mock_env_fixture


@pytest.fixture
def mock_bstep():
    """Mock batch stepper class with fixed run_episode_batch return."""
    bstep_cls = mock.create_autospec(batch_steppers.LocalBatchStepper)
    bstep_cls.return_value.run_episode_batch.return_value = (
        testing.construct_episodes(
            actions=[
                [0, 2], [0, 2], [0, 2],  # Three first episodes action 0
                [1, 2], [1, 2], [1, 2],  # Three last episodes action 1
            ],
            rewards=[
                [0, 1], [0, 1], [0, 1],  # Higher mean return, action 0
                [0, 0], [0, 0], [0, 2],  # Higher max return, action 1
            ], truncated=True))
    return bstep_cls


def test_number_of_simulations(mock_env, mock_bstep):
    # Set up
    n_rollouts = 7
    n_envs = 2
    agent = agents.ShootingAgent(
        batch_stepper_class=mock_bstep,
        n_rollouts=n_rollouts,
        n_envs=n_envs
    )

    # Run
    observation = mock_env.reset()
    testing.run_with_dummy_network_response(
        agent.reset(mock_env, observation)
    )
    testing.run_without_suspensions(agent.act(None))

    # Test
    assert mock_bstep.return_value.run_episode_batch.call_count == \
        math.ceil(n_rollouts / n_envs)


@pytest.mark.parametrize('aggregate_fn,x_action',
                         [(np.mean, 0),
                          (np.max, 1)])
def test_greedy_decision_for_all_aggregators(mock_env, mock_bstep,
                                             aggregate_fn, x_action):
    # Set up
    agent = agents.ShootingAgent(
        aggregate_fn=aggregate_fn,
        batch_stepper_class=mock_bstep,
        n_rollouts=1,
    )
    x_onehot_action = np.zeros(3)
    x_onehot_action[x_action] = 1

    # Run
    observation = mock_env.reset()
    testing.run_with_dummy_network_response(
        agent.reset(mock_env, observation)
    )
    (actual_action, agent_info) = testing.run_without_suspensions(
        agent.act(None)
    )

    # Test
    assert actual_action == x_action
    np.testing.assert_array_equal(agent_info['action_histogram'],
                                  x_onehot_action)


@pytest.mark.parametrize('estimate_fn,x_action,logits',
                         [(mc_simulation.truncated_return, 0,
                           None),  # No predictions.
                          (mc_simulation.bootstrap_return_with_value, 1,
                           # The first three predictions are for the first
                           # action. No extra "bonus".
                           # The last three predictions are for the second
                           # action. Extra bonus of 1 for mean aggregator.
                           (np.array([[0]] * 3 + [[1]] * 3), None))
                          ])
def test_greedy_decision_for_all_return_estimators(mock_env, mock_bstep,
                                                   estimate_fn, x_action,
                                                   logits):
    # Set up
    agent = agents.ShootingAgent(
        estimate_fn=estimate_fn,
        batch_stepper_class=mock_bstep,
        n_rollouts=1,
    )
    x_onehot_action = np.zeros(3)
    x_onehot_action[x_action] = 1

    # Run
    observation = mock_env.reset()
    testing.run_with_dummy_network_response(
        agent.reset(mock_env, observation)
    )
    (actual_action, agent_info) = testing.run_with_constant_network_prediction(
        agent.act(None),
        logits=logits
    )

    # Test
    assert actual_action == x_action
    np.testing.assert_array_equal(agent_info['action_histogram'],
                                  x_onehot_action)


@pytest.mark.parametrize('rollout_time_limit', [None, 7])
def test_rollout_time_limit(mock_env, rollout_time_limit):
    # Set up
    rollout_max_len = 10  # It must be greater then rollout_time_limit!
    mock_env.action_space.sample.return_value = 0
    mock_env.step.side_effect = \
        [('d', 0, False, {})] * (rollout_max_len - 1) + [('d', 0, True, {})]
    mock_env.clone_state.return_value = 's'
    mock_env.restore_state.return_value = 'o'

    if rollout_time_limit is None:
        x_rollout_time_limit = rollout_max_len
    else:
        x_rollout_time_limit = rollout_time_limit

    def _estimate_fn(episodes, discount):
        del discount
        # Test
        for episode in episodes:
            actual_rollout_time_limit = len(episode.transition_batch.done)
            assert actual_rollout_time_limit == x_rollout_time_limit

        return [1.] * len(episodes)

        # To indicate it's a coroutine:
        yield  # pylint: disable=unreachable

    with mock.patch('alpacka.agents.mc_simulation.type') as mock_type:
        mock_type.return_value = lambda: mock_env
        agent = agents.ShootingAgent(
            n_rollouts=1,
            rollout_time_limit=rollout_time_limit,
            estimate_fn=_estimate_fn,
            n_envs=1,
        )

        # Run
        observation = mock_env.reset()
        testing.run_with_dummy_network_response(
            agent.reset(mock_env, observation)
        )
        testing.run_without_suspensions(agent.act(None))
