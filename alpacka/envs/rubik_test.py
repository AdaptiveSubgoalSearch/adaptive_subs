"""Tests for alpacka.envs.RubiksCube."""

import itertools

import numpy as np
import pytest

from alpacka import envs


n = 3
n_shuffles = 100


@pytest.fixture
def env():
    return envs.RubiksCube(n=n, n_shuffles=n_shuffles, smart_shuffles=True)


@pytest.mark.parametrize('_', range(10))
def test_nonredundant_shuffles(env, _):
    # Smart 2-shuffled cube cannot be already solved.

    # Set up
    env.n_shuffles = 2

    # Run
    observation = env.reset()
    goal_state = env.get_goal_state()

    # Test
    np.testing.assert_equal(np.all(observation == goal_state), False)


@pytest.mark.parametrize('action', range(12))
def test_reversible_action(env, action):
    # Any action can be reversed by a complementary one.

    # Run
    initial_observation = env.reset()
    env.step(action)
    reverse_action = env.reverse_action(action)
    final_observation, _, _, _ = env.step(reverse_action)

    # Test
    np.testing.assert_array_equal(initial_observation, final_observation)


@pytest.mark.parametrize('action', range(12))
def test_full_rotation(env, action):
    # Any action can be reversed by repeating it 4 times.

    # Run
    initial_observation = env.reset()
    for _ in range(4):
        final_observation, _, _, _ = env.step(action)

    # Test
    np.testing.assert_array_equal(initial_observation, final_observation)


def test_checkerboard_template(env):
    # Make checkerboard pattern and compare with template.

    # Set up
    env.n_shuffles = 0

    checkerboard = []
    for face in range(6):
        opp = face - (face % 2) + (1 - face % 2)
        checkerboard += [face, opp] * 5
        checkerboard = checkerboard[:-1]
    checkerboard = np.eye(6)[checkerboard]

    # Run
    env.reset()
    for action in [0, 2, 4, 6, 8, 10]:  # rotate each face twice
        env.step(action)
        observation, _, _, _ = env.step(action)

    checkerboard = checkerboard.reshape(observation.shape)

    # Test
    np.testing.assert_array_equal(observation, checkerboard)


@pytest.mark.parametrize('order', itertools.permutations([0, 1, 2]))
def test_checkerboard_order(env, order):
    # Make checkerboard pattern, fairly regardless of the moves order.

    # Set up
    env.n_shuffles = 0

    env.reset()
    for action in [0, 2, 4, 6, 8, 10]:  # rotate each face twice
        env.step(action)
        observation, _, _, _ = env.step(action)
    checkerboard = observation

    env.reset()

    # Run
    for axis in order:
        action1 = 4 * axis
        action2 = 4 * axis + 2
        moves = [action1, action1, action2, action2]
        np.random.shuffle(moves)

        for action in moves:
            observation, _, _, _ = env.step(action)

    # Test
    np.testing.assert_array_equal(observation, checkerboard)
