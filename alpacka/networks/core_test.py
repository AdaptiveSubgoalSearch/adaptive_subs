"""Tests for alpacka.networks.core."""

import functools
from unittest import mock

import gym
import numpy as np
import pytest

from alpacka import data
from alpacka import testing
from alpacka.agents import dummy
from alpacka.networks import core
from alpacka.networks import keras


_OtherRequest = data.register_prediction_request('OtherRequest')  # pylint: disable=invalid-name


def test_union_network_basic_usage():
    # Set agent network up
    agent_network_mock = mock.Mock()
    agent_network_mock.predict = mock.Mock(return_value='agent')
    agent_input_shape = (3, 2)
    agent_network_sig = data.NetworkSignature(
        input=data.TensorSignature(agent_input_shape),
        output=data.TensorSignature((1,))
    )

    # Set model network up
    model_network_mock = mock.Mock()
    model_network_mock.predict = mock.Mock(return_value='model')
    model_input_shape = (4, 5)
    model_network_sig = data.NetworkSignature(
        input=data.TensorSignature(model_input_shape),
        output=data.TensorSignature((3, 6))
    )

    # Set UnionNetwork up
    def get_network_fn(network_mock, expected_network_sig):
        def network_fn(network_signature):
            assert network_signature == expected_network_sig
            return network_mock
        return network_fn

    union_network_sig = {
        data.AgentRequest: agent_network_sig,
        data.ModelRequest: model_network_sig,
    }

    network = core.UnionNetwork(union_network_sig, request_to_network={
        data.AgentRequest: get_network_fn(
            agent_network_mock, agent_network_sig
        ),
        data.ModelRequest: get_network_fn(
            model_network_mock, model_network_sig
        ),
    })

    # Check positive cases
    assert network.predict(
        data.AgentRequest(np.zeros(agent_input_shape))
    ) == 'agent'
    assert network.predict(
        data.ModelRequest(np.zeros(model_input_shape))
    ) == 'model'

    # Check wrong cases
    with pytest.raises(KeyError):
        # No network specified for _OtherRequest
        network.predict(_OtherRequest(np.zeros(agent_input_shape)))
    with pytest.raises(ValueError):
        # Request type not created by data.register_prediction_request()
        network.predict(np.zeros(agent_input_shape))


def test_union_network_arguments_mismatch():
    sample_signature = data.NetworkSignature(
        input=data.TensorSignature((2, 2)),
        output=data.TensorSignature((1,)),
    )
    network_signature = {
        data.AgentRequest: sample_signature,
        _OtherRequest: sample_signature,
    }

    with pytest.raises(ValueError):
        core.UnionNetwork(network_signature, request_to_network={
            data.AgentRequest: lambda x: mock.Mock(),
            data.ModelRequest: lambda x: mock.Mock(),
        })


def test_try_register_colliding_prediction_requests():
    # Please note that this test modifies global state, as it effectively
    # registers some gin configurables.

    with pytest.raises(ValueError):
        # Collides with existing request type data.AgentRequest
        data.register_prediction_request('AgentRequest')

    data.register_prediction_request('SomeRequest')
    with pytest.raises(ValueError):
        data.register_prediction_request('SomeRequest')

    data.register_prediction_request(
        'AnotherRequest', module='alpacka.networks.core_test'
    )
    with pytest.raises(ValueError):
        data.register_prediction_request(
            'AnotherRequest', module='alpacka.data'
        )


def test_union_network_prediction():
    # Spaces as for Sokoban
    observation_space = gym.spaces.Box(0, 1, shape=(8, 8, 7), dtype=np.float32)
    action_space = gym.spaces.Discrete(4)
    mock_env = mock.Mock()
    mock_env.action_space = action_space

    # Set agent up
    agent = dummy.DummyTwoNetworkAgent(random_order=False)
    # Agent needs to get mock_env.action_space information
    testing.run_without_suspensions(agent.reset(mock_env, None))
    network_signature = agent.network_signature(
        observation_space, action_space
    )

    # Set network up
    network = core.UnionNetwork(network_signature, request_to_network={
        data.AgentRequest: functools.partial(
            keras.KerasNetwork, model_fn=keras.convnet_mnist
        ),
        data.ModelRequest: functools.partial(
            keras.KerasNetwork, model_fn=keras.fcn_for_env_model
        ),
    })

    sample_observation = np.zeros(
        observation_space.shape, dtype=observation_space.dtype
    )

    # Check that agent works with network without any exceptions being raised
    (action, _) = testing.run_with_network(
        agent.act(sample_observation), network
    )
    assert action_space.contains(action)


def test_union_network_save_restore(tmp_path):
    checkpoint_path = tmp_path / 'checkpoint'

    network_signature = {
        data.AgentRequest: data.NetworkSignature(
            input=data.TensorSignature((2, 2)),
            output=data.TensorSignature((1,)),
        )
    }
    union_network = core.UnionNetwork(network_signature, request_to_network={
        data.AgentRequest: core.DummyNetwork,
    })
    subnetwork = next(iter(union_network.subnetworks.values()))

    # Set the subnetwork parameters to one value.
    subnetwork.params = 'foo'

    # Save the UnionNetwork.
    union_network.save(checkpoint_path)

    # Set the subnetwork parameters to another value.
    subnetwork.params = 'bar'
    assert subnetwork.params == 'bar'

    # Restore the UnionNetwork.
    union_network.restore(checkpoint_path)

    # Assert that the original parameters were restored.
    assert subnetwork.params == 'foo'
