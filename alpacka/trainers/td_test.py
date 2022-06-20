"""Tests for alpacka.trainers.td."""

import collections
import os

import numpy as np

from alpacka import data
from alpacka.networks import core
from alpacka.trainers import td


def test_target_n_return():
    """A rudimentary test of target_n_return."""

    transition_batch = data.Transition(
        observation=np.expand_dims(np.arange(10), 1),
        action='not used',
        reward=np.arange(10),
        done=[False] * 10,
        next_observation=np.expand_dims(np.arange(1, 11), 1),
        agent_info='not used',
    )

    e = data.Episode(
        transition_batch=transition_batch,
        return_='not used',
        solved='not used',
        truncated=False,
    )

    datapoints = td.target_n_return(e, 1, 0.99)

    bootstrap_gamma = np.full((10, 1), 0.99)
    bootstrap_gamma[9, 0] = 0.0
    bootstrap_obs = np.expand_dims(np.arange(1, 11), 1)
    cum_reward = np.expand_dims(np.arange(10), 1)

    assert np.array_equal(datapoints.bootstrap_gamma, bootstrap_gamma)
    assert np.array_equal(datapoints.bootstrap_obs, bootstrap_obs)
    assert np.array_equal(datapoints.cum_reward, cum_reward)


def test_save_restore(tmpdir):
    TestTransition = collections.namedtuple(
        'TestTransition', ['observation', 'reward', 'next_observation']
    )

    network_sig = data.NetworkSignature(
        input=data.TensorSignature(shape=()),
        output=data.TensorSignature(shape=(1,)),
    )
    trainer_fn = lambda: td.TDTrainer(
        network_signature=network_sig,
        temporal_diff_n=1,
        batch_size=1,
        n_steps_per_epoch=1,
        replay_buffer_capacity=1,
    )

    # Dump the Trainer with one episode.
    orig_trainer = trainer_fn()
    orig_trainer.add_episode(data.Episode(
        transition_batch=TestTransition(
            observation=np.zeros(1),
            reward=np.arange(1),
            next_observation=np.zeros(1),
        ),
        return_=123,
        solved=False,
    ))
    trainer_path = os.path.join(tmpdir, 'trainer')
    orig_trainer.save(trainer_path)

    # Check that we can restore and train for one epoch.
    restored_trainer = trainer_fn()
    restored_trainer.restore(trainer_path)

    class _TestNetwork(core.DummyNetwork):

        def train(self, data_stream, n_steps):
            np.testing.assert_equal(
                list(data_stream()),
                [data.zero_pytree(
                    (network_sig.input, network_sig.output),
                    shape_prefix=(1,)
                ) + data.one_pytree(
                    (network_sig.output,), shape_prefix=(1,)
                )],
            )
            return {}

    restored_trainer.train_epoch(_TestNetwork(network_sig))
