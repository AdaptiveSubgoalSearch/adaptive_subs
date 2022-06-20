"""Tests for alpacka.utils.metric."""

import numpy as np

from alpacka import testing
from alpacka.utils import metric as metric_utils


def test_compute_episode_metrics():
    # Set up
    rewards = [
        [1, 2, 3, 4, 5],
        [6, 7, 8],
        [9, 0, 10, 11, 12, 13, 14, 15, 16, 17],
        [-3, -2, -1, 0, 1, 2, 3],
        [10],
    ]
    returns = [
        sum(episode_rewards) for episode_rewards in rewards
    ]
    lengths = [
        len(episode_rewards) for episode_rewards in rewards
    ]
    episodes = testing.construct_episodes(actions=rewards,
                                          rewards=rewards,
                                          solved=True)

    # Run
    metrics = metric_utils.compute_episode_metrics(episodes)

    # Test
    np.testing.assert_almost_equal(
        metrics['return_mean'], np.mean(returns))
    np.testing.assert_almost_equal(
        metrics['return_median'], np.median(returns))
    np.testing.assert_almost_equal(
        metrics['return_std'], np.std(returns, ddof=1))
    np.testing.assert_almost_equal(
        metrics['length_mean'], np.mean(lengths))
    np.testing.assert_almost_equal(
        metrics['length_median'], np.median(lengths))
    np.testing.assert_almost_equal(
        metrics['length_std'], np.std(lengths, ddof=1))
    np.testing.assert_almost_equal(
        metrics['solved_rate'], 1.0)
