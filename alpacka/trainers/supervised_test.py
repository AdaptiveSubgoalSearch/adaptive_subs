"""Tests for alpacka.trainers.supervised."""

import collections
import os

import numpy as np
import pytest

from alpacka import data
from alpacka.networks import core
from alpacka.networks import keras
from alpacka.trainers import supervised


class _TestNetwork(core.DummyNetwork):

    def __init__(self, signature):
        super().__init__(signature)
        self._signature = signature

    def train(self, data_stream, n_steps):
        np.testing.assert_equal(
            list(data_stream()),
            [data.zero_pytree(
                (self._signature.input, self._signature.output),
                shape_prefix=(1,)
            ) + data.one_pytree(
                (self._signature.output,), shape_prefix=(1,)
            )],
        )
        return {}


def test_integration_with_keras():
    TestTransition = collections.namedtuple('TestTransition', ['observation'])

    # Just a smoke test, that nothing errors out.
    n_transitions = 10
    obs_shape = (4,)
    network_sig = data.NetworkSignature(
        input=data.TensorSignature(shape=obs_shape),
        output=data.TensorSignature(shape=(1,)),
    )
    trainer = supervised.SupervisedTrainer(
        network_signature=network_sig,
        target=supervised.target_solved,
        batch_size=2,
        n_steps_per_epoch=3,
        replay_buffer_capacity=n_transitions,
    )
    trainer.add_episode(data.Episode(
        transition_batch=TestTransition(
            observation=np.zeros((n_transitions,) + obs_shape),
        ),
        return_=123,
        solved=False,
    ))
    network = keras.KerasNetwork(network_signature=network_sig)
    trainer.train_epoch(network)


def test_multiple_targets():
    TestTransition = collections.namedtuple(
        'TestTransition', ['observation', 'agent_info']
    )

    network_sig = data.NetworkSignature(
        input=data.TensorSignature(shape=(1,)),
        # Two outputs.
        output=(
            data.TensorSignature(shape=(1,)),
            data.TensorSignature(shape=(2,)),
        ),
    )
    trainer = supervised.SupervisedTrainer(
        network_signature=network_sig,
        # Two targets.
        target=(supervised.target_solved, supervised.target_qualities),
        batch_size=1,
        n_steps_per_epoch=1,
        replay_buffer_capacity=1,
    )
    trainer.add_episode(data.Episode(
        transition_batch=TestTransition(
            observation=np.zeros((1, 1)),
            agent_info={'qualities': np.zeros((1, 2))},
        ),
        return_=123,
        solved=False,
    ))

    # _TestNetwork asserts the shape of training batches.
    trainer.train_epoch(_TestNetwork(network_sig))


def test_target_signature_check():
    TestTransition = collections.namedtuple(
        'TestTransition', ['observation', 'agent_info']
    )

    network_sig = data.NetworkSignature(
        input=data.TensorSignature(shape=(1,)),
        # Output is a 2-vector.
        output=data.TensorSignature(shape=(2,)),
    )
    trainer = supervised.SupervisedTrainer(
        network_signature=network_sig,
        # Target is a singleton vector.
        target=supervised.target_value,
        batch_size=1,
        n_steps_per_epoch=1,
        replay_buffer_capacity=1,
    )

    # This should error out because of a shape mismatch.
    with pytest.raises(ValueError):
        trainer.add_episode(data.Episode(
            transition_batch=TestTransition(
                observation=np.zeros((1, 1)),
                agent_info={'value': np.zeros((1,))},
            ),
            return_=123,
            solved=False,
        ))


def test_save_restore(tmpdir):
    TestTransition = collections.namedtuple('TestTransition', ['observation'])

    # Just a smoke test, that nothing errors out.
    network_sig = data.NetworkSignature(
        input=data.TensorSignature(shape=()),
        output=data.TensorSignature(shape=(1,)),
    )
    trainer_fn = lambda: supervised.SupervisedTrainer(
        network_signature=network_sig,
        target=supervised.target_solved,
        batch_size=1,
        n_steps_per_epoch=1,
        replay_buffer_capacity=1,
    )

    orig_trainer = trainer_fn()
    orig_trainer.add_episode(data.Episode(
        transition_batch=TestTransition(observation=np.zeros(1)),
        return_=123,
        solved=False,
    ))
    trainer_path = os.path.join(tmpdir, 'trainer')
    orig_trainer.save(trainer_path)

    restored_trainer = trainer_fn()
    restored_trainer.restore(trainer_path)

    restored_trainer.train_epoch(_TestNetwork(network_sig))
