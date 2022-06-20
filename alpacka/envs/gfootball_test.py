"""Tests for alpacka.envs.gfootball."""

import numpy as np
import pytest

from alpacka.envs import gfootball


football_test = pytest.mark.skipif(
    gfootball.football_env is None,
    reason='Could not import Google Research Football.',
)


@football_test
def test_rerun_plan_after_restore_yields_the_same_trajectory_in_grf():
    # Set up
    env = gfootball.GoogleFootball()
    env.reset()
    state = env.clone_state()
    plan = (4, 4, 5, 5, 6, 6, 5, 5, 4, 4, 5, 12, 0, 0, 0, 0)

    # Run
    trajectories = [[], []]
    for idx in range(2):
        env.restore_state(state)
        for act in plan:
            obs, _, _, _ = env.step(act)
            trajectories[idx].append(obs)

    # Test
    first = np.array(trajectories[0])
    second = np.array(trajectories[1])
    assert np.array_equal(first, second)


@football_test
def test_rerun_plan_after_reset_yields_different_trajectory_in_grf():
    # Set up
    env = gfootball.GoogleFootball()
    env.reset()
    plan = (4, 4, 5, 5, 6, 6, 5, 5, 4, 4, 5, 12, 0, 0, 0, 0)

    # Run
    trajectories = [[], []]
    for idx in range(2):
        env.reset()
        for act in plan:
            obs, _, _, _ = env.step(act)
            trajectories[idx].append(obs)

    # Test
    first = np.array(trajectories[0])
    second = np.array(trajectories[1])
    assert not np.array_equal(first, second)


@football_test
def test_restore_after_done_in_grf():
    env = gfootball.GoogleFootball(env_name='academy_empty_goal_close')
    env.reset()
    state = env.clone_state()
    # Go right until reaching the goal.
    done = False
    while not done:
        (_, _, done, _) = env.step(5)
    env.restore_state(state)
    env.step(5)  # Check if we can make a step after restore.
