"""Test trainer for the multiple network setup."""

import functools

import numpy as np

from alpacka import data
from alpacka import networks
from alpacka.networks import keras
from alpacka.trainers import dummy
from alpacka.trainers import supervised
from alpacka.trainers import union


def test_integration_with_union_network():
    """Smoke test to check that UnionTrainer works with UnionNetwork.

    Populates UnionTrainer with dummy transitions and trains UnionNetwork.
    The test excercises a sample setup for OnlineAgent with trainable model.
    """

    # Specify data shape.
    n_transitions = 10
    obs_shape = (4, 2, 3)
    action_space_size = 2
    network_signature = {
        data.AgentRequest: data.NetworkSignature(
            input=data.TensorSignature(shape=obs_shape),
            output=data.TensorSignature(shape=(1,)),
        ),
        data.ModelRequest: data.NetworkSignature(
            input={
                'observation': data.TensorSignature(shape=obs_shape),
                'action': data.TensorSignature(shape=(action_space_size,)),
            },
            output={
                'next_observation': data.TensorSignature(shape=obs_shape),
                'reward': data.TensorSignature(shape=(1,)),
                'done': data.TensorSignature(shape=(1,)),
            }
        ),
    }

    # Set UnionNetwork up.
    network = networks.core.UnionNetwork(
        network_signature=network_signature,
        request_to_network={
            data.AgentRequest: functools.partial(
                keras.KerasNetwork, model_fn=keras.convnet_mnist
            ),
            data.ModelRequest: functools.partial(
                keras.KerasNetwork, model_fn=keras.fcn_for_env_model
            ),
        }
    )

    # Set UnionTrainer up.
    common_trainer_kwargs = dict(
        batch_size=2,
        n_steps_per_epoch=3,
        replay_buffer_capacity=n_transitions,
    )
    agent_trainer_fn = functools.partial(
        supervised.SupervisedTrainer,
        input=supervised.input_observation,
        target=supervised.target_value,
        **common_trainer_kwargs,
    )
    model_trainer_fn = functools.partial(
        supervised.SupervisedTrainer,
        input={
            'observation': supervised.input_observation,
            'action': supervised.input_action,
        },
        target={
            'next_observation': supervised.target_next_observation,
            'reward': supervised.target_reward,
            'done': supervised.target_done,
        },
        **common_trainer_kwargs,
    )
    trainer = union.UnionTrainer(
        network_signature=network_signature,
        request_to_trainer={
            data.AgentRequest: agent_trainer_fn,
            data.ModelRequest: model_trainer_fn,
        }
    )

    # Add transitions to UnionTrainer.
    obs_batch_shape = (n_transitions,) + obs_shape
    scalar_batch_shape = (n_transitions,)
    trainer.add_episode(data.Episode(
        transition_batch=data.Transition(
            agent_info={'value': np.zeros(scalar_batch_shape)},

            observation=np.zeros(obs_batch_shape),
            action=np.zeros(scalar_batch_shape, dtype=np.int32),

            next_observation=np.zeros(obs_batch_shape),
            reward=np.zeros(scalar_batch_shape),
            done=np.zeros(scalar_batch_shape),
        ),
        return_=123,
        solved=False,
        action_space_size=action_space_size,
    ))

    # Train the network.
    # Check that training works without any exception being raised.
    trainer.train_epoch(network)


def test_save_restore(tmp_path):
    checkpoint_path = tmp_path / 'checkpoint'

    network_signature = {
        data.AgentRequest: data.NetworkSignature(
            input=data.TensorSignature((2, 2)),
            output=data.TensorSignature((1,)),
        )
    }
    union_trainer = union.UnionTrainer(network_signature, request_to_trainer={
        data.AgentRequest: dummy.DummyTrainer,
    })
    subtrainer = next(iter(union_trainer.subtrainers.values()))

    # Set the subtrainer replay buffer to one value.
    subtrainer.replay_buffer = 'foo'

    # Save the UnionNetwork.
    union_trainer.save(checkpoint_path)

    # Set the subtrainer replay buffer to another value.
    subtrainer.replay_buffer = 'bar'
    assert subtrainer.replay_buffer == 'bar'

    # Restore the UnionNetwork.
    union_trainer.restore(checkpoint_path)

    # Assert that the original parameters were restored.
    assert subtrainer.replay_buffer == 'foo'
