"""Utilities for gym spaces."""

import gym

from alpacka import data


def element_iter(action_space):
    """Returns an iterator over elements of a gym space."""
    try:
        return iter(action_space)
    except TypeError:
        if isinstance(action_space, gym.spaces.Discrete):
            return iter(range(action_space.n))
        else:
            raise TypeError('Space {} does not support iteration.'.format(
                type(action_space)
            ))


def signature(space):
    """Returns a TensorSignature of elements of the given space."""
    return data.TensorSignature(shape=space.shape, dtype=space.dtype)


def max_size(space):
    """Returns the maximum number of elements in the space."""
    try:
        return space.max_size
    except AttributeError:
        if isinstance(space, gym.spaces.Discrete):
            return space.n
        else:
            raise TypeError('Space {} is not countable.'.format(
                type(space)
            ))
