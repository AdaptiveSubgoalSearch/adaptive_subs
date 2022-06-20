"""Data transformation utils."""

import numpy as np
import scipy.signal


def discount_cumsum(x, discount):
    """Magic from rllab for computing discounted cumulative sums of vectors.

    Args:
        x (np.array): sequence of floats (eg. rewards from a single episode in
            RL settings)
        discount (float): discount factor (in RL known as gamma)

    Returns:
        Array of cumulative discounted sums. For example:

        If vector x has a form
            [x0,
             x1,
             x2]

        Then the output would be:
            [x0 + discount * x1 + discount^2 * x2,
             x1 + discount * x2,
             x2]
    """
    return scipy.signal.lfilter(
        [1], [1, float(-discount)], x[::-1], axis=0
    )[::-1]


def one_hot_encode(values, value_space_size, dtype=np.float32):
    """One-hot encodes 1D batch of values.

    Args:
        values (array_like): One dimensional list/array of integers to encode.
        value_space_size (int): Number of possible values.
        dtype (np.dtype): Type of resulting array.

    Returns:
        2D numpy array of shape (len(values), value_space_size)
    """
    target_shape = (len(values), value_space_size)

    result = np.zeros(target_shape, dtype=dtype)
    result[np.arange(target_shape[0]), values] = 1

    return result


def map_dict_keys(input_dict, mapper):
    """Creates a new dict with modified keys but the same values.

    Args:
        input_dict: dict(key1 -> value)
        mapper: callable(key1 -> key2)

    Raises:
        ValueError: on collisions of keys

    Returns:
        dict(key2 -> value)
    """
    result = {
        mapper(key): value
        for key, value in input_dict.items()
    }
    if len(result.keys()) != len(input_dict.keys()):
        raise ValueError(
            'There are collisions of keys after applying the mapper.'
        )
    return result


def zip_dicts_strict(dict1, dict2):
    """Zips two dicts.

    Checks that both dicts have the same set of keys.
    """
    if dict1.keys() != dict2.keys():
        raise ValueError(
            'Keys of dicts do not match.\n'
            f'{dict1.keys()} != {dict2.keys()}'
        )

    return {
        key: (val1, dict2[key])
        for key, val1 in dict1.items()
    }
