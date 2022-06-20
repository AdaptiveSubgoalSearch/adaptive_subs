"""Helper functions for metrics calculation."""

import numpy as np


def compute_episode_metrics(episodes):
    """Computes metrics for a list of episodes.

    Args:
        episodes (list): List of data.Episode objects.

    Returns:
        dict: Keys are metric names and values are the metrics.
    """
    returns_ = np.array([episode.return_ for episode in episodes])
    lengths = np.array([episode.transition_batch.reward.shape[0]
                        for episode in episodes])

    solved_rate = sum(
        int(episode.solved) for episode in episodes
        if episode.solved is not None
    ) / len(episodes)

    return dict(
        return_mean=np.mean(returns_),
        return_median=np.median(returns_),
        return_std=np.std(returns_, ddof=1),
        length_mean=np.mean(lengths),
        length_median=np.median(lengths),
        length_std=np.std(lengths, ddof=1),
        solved_rate=solved_rate,
    )


def compute_scalar_statistics(x, prefix=None, with_min_and_max=False):
    """Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x (np.ndarray): Samples of the scalar to produce statistics for.
        prefix (str): Prefix to put before a statistic name, separated with
            an underscore.
        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.

    Note:
        NaN values are ignored when calculating statistics.

    Returns:
        dict: Statistic names as keys (can be prefixed, see the prefix
        argument) and statistics as values.
    """
    prefix = prefix + '_' if prefix else ''
    stats = {}

    stats[prefix + 'mean'] = np.nanmean(x)
    stats[prefix + 'std'] = np.nanstd(x)
    if with_min_and_max:
        stats[prefix + 'min'] = np.nanmin(x)
        stats[prefix + 'max'] = np.nanmax(x)

    return stats
