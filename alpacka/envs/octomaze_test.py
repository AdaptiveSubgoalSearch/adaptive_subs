"""Tests for alpacka.envs.Octomaze."""

import numpy as np
import pytest

from alpacka import envs


num_corridors = 4
corridor_length = 5


@pytest.fixture
def env():
    return envs.Octomaze(num_corridors, corridor_length)


@pytest.mark.parametrize('action', range(num_corridors))
def test_pick_corridor(env, action):
    # Set up
    init_observation = np.zeros([num_corridors, corridor_length])
    target_observation = np.zeros([num_corridors, corridor_length])
    target_observation[action][0] = 1

    # Run
    observation = env.reset()
    # Pick a corridor.
    next_observation, reward, done, info = env.step(action)

    # Test
    np.testing.assert_array_equal(observation, init_observation)
    np.testing.assert_array_equal(next_observation, target_observation)
    np.testing.assert_equal(reward, 0)
    np.testing.assert_equal(done, False)
    np.testing.assert_equal(info, {'solved': False})


@pytest.mark.parametrize('action', range(num_corridors))
def test_step_corridor(env, action):
    # Set up
    init_observation = np.zeros([num_corridors, corridor_length])
    target_observation = np.zeros([num_corridors, corridor_length])
    target_observation[action][1] = 1

    # Run
    observation = env.reset()
    # Pick a corridor.
    env.step(action)
    # Step in the corridor.
    next_observation, reward, done, info = env.step(action)

    # Test
    np.testing.assert_array_equal(observation, init_observation)
    np.testing.assert_array_equal(next_observation, target_observation)
    np.testing.assert_equal(reward, 0)
    np.testing.assert_equal(done, False)
    np.testing.assert_equal(info, {'solved': False})


@pytest.mark.parametrize('action', range(num_corridors))
def test_change_corridor(env, action):
    # Set up
    init_observation = np.zeros([num_corridors, corridor_length])
    target_observation = np.zeros([num_corridors, corridor_length])
    target_observation[action][0] = 1  # Should stay in the chosen corridor.

    # Run
    observation = env.reset()
    # Pick a corridor.
    env.step(action)
    # Try to step in the next corridor.
    next_observation, reward, done, info = env.step(
        (action + 1) % num_corridors)

    # Test
    np.testing.assert_array_equal(observation, init_observation)
    np.testing.assert_array_equal(next_observation, target_observation)
    np.testing.assert_equal(reward, 0)
    np.testing.assert_equal(done, False)
    np.testing.assert_equal(info, {'solved': False})


@pytest.mark.parametrize('action', range(num_corridors))
def test_traverse_corridor(env, action):
    # Set up
    init_observation = np.zeros([num_corridors, corridor_length])
    target_observation = np.ones([num_corridors, corridor_length])

    # Run
    observation = env.reset()
    # Pick a corridor and go through it.
    for _ in range(corridor_length + 1):
        next_observation, reward, done, info = env.step(action)

    # Test
    np.testing.assert_array_equal(observation, init_observation)
    np.testing.assert_array_equal(next_observation, target_observation)
    np.testing.assert_equal(reward, 1)
    np.testing.assert_equal(done, True)
    np.testing.assert_equal(info, {'solved': True})
