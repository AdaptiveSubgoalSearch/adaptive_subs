"""Ops for manipulating pytrees.

Pytrees are nested structures of lists/tuples/namedtuples/dicts. They typically
have numpy arrays at leaves.

Behavior of those ops is well described in alpacka.data.ops_test.
"""

import functools

import numpy as np


_leaf_types = set()


def register_leaf_type(leaf_type):
    """Registers a leaf type.

    Leaf type is a type that should not be recursed into in nested_* functions.
    Useful for defining namedtuples such as data.TensorSignature.

    Args:
        leaf_type (type): Type to register as a leaf.
    """
    _leaf_types.add(leaf_type)


def _is_leaf(x):
    """Returns whether pytree is a leaf."""
    if type(x) in _leaf_types:  # pylint: disable=unidiomatic-typecheck
        return True
    return not isinstance(x, (tuple, list, dict))


def _is_namedtuple_instance(x):
    """Determines if x is an instance of a namedtuple type."""
    if isinstance(x, tuple):
        return hasattr(x, '_fields')
    else:
        return False


def _verbose(arg_index):
    """Makes a nested_* function log its input type on error."""
    def decorator(wrapped):
        @functools.wraps(wrapped)
        def wrapper(*args, **kwargs):
            try:
                return wrapped(*args, **kwargs)
            except Exception as e:
                # Augment the exception with information about the pytree type.
                try:
                    # Only do that for exceptions that carry string messages.
                    (msg,) = e.args
                    msg += '\n  {} input pytree type:\n    {}'.format(
                        wrapped.__name__,
                        # Unwrap nested_map to avoid infinite recursion.
                        nested_map.__wrapped__(type, args[arg_index]),
                    )
                    e.args = (msg,)
                except Exception:  # pylint: disable=broad-except
                    # If that fails, it means that the exception carries
                    # something different/more than a string message. Just
                    # re-raise the original exception.
                    pass
                raise e
        return wrapper
    return decorator


@_verbose(arg_index=1)
def nested_map(f, x, stop_fn=_is_leaf):
    """Maps a function through a pytree.

    Args:
        f (callable): Function to map.
        x (any): Pytree to map over.
        stop_fn (callable): Optional stopping condition for the recursion. By
            default, stops on leaves.

    Returns:
        pytree: Result of mapping.
    """
    if stop_fn(x):
        return f(x)

    if _is_namedtuple_instance(x):
        return type(x)(*nested_map.__wrapped__(f, tuple(x), stop_fn=stop_fn))
    if isinstance(x, dict):
        return {
            k: nested_map.__wrapped__(f, v, stop_fn=stop_fn)
            for (k, v) in x.items()
        }
    assert isinstance(x, (list, tuple)), (
        'Non-exhaustive pattern match for {}.'.format(type(x))
    )
    return type(x)(nested_map.__wrapped__(f, y, stop_fn=stop_fn) for y in x)


def _assert_zippable(xs):
    assert not _is_leaf(xs), f'Cannot zip a leaf: {xs}.'
    assert xs, f'Cannot zip an empty sequence: {xs}.'

    class Leaf:
        pass

    def extract_structure(x):
        # Step 1: forget all information about the leaves.
        x = nested_map(lambda _: Leaf, x)

        # Step 2: convert lists to tuples.
        def list_to_tuple(x):
            if isinstance(x, list):
                x = tuple(x)
            return x

        return nested_reduce(list_to_tuple, x, to_list=False)

    # Assert that the structures are the same. If not, print out the nested
    # types, as they carry more information.
    structures = extract_structure(xs)
    nested_types = nested_map(type, xs)
    assert all(structure == structures[0] for structure in structures), (
        f'Cannot zip incompatible pytrees: {nested_types}.'
    )


@_verbose(arg_index=0)
def nested_zip(xs):
    """Zips a list of pytrees.

    Inverse of nested_unzip.

    Example:
        nested_zip([((1, 2), 3), ((4, 5), 6)]) == (([1, 4], [2, 5]), [3, 6])
    """
    _assert_zippable(xs)

    if _is_leaf(xs[0]):
        return xs

    if _is_namedtuple_instance(xs[0]):
        return type(xs[0])(*nested_zip.__wrapped__([tuple(x) for x in xs]))
    elif isinstance(xs[0], (list, tuple)):
        for x in xs:
            assert len(x) == len(xs[0]), (
                'Cannot zip sequences of different lengths: '
                '{} and {}'.format(len(x), len(xs[0]))
            )

        return type(xs[0])(
            nested_zip.__wrapped__([x[i] for x in xs])
            for i in range(len(xs[0]))
        )
    elif isinstance(xs[0], dict):
        return {
            k: nested_zip.__wrapped__([x[k] for x in xs])
            for k in xs[0].keys()
        }
    else:
        raise TypeError(
            'Non-exhaustive pattern match for {}.'.format(type(xs[0]))
        )


def _is_last_level(x):
    """Returns whether pytree is at the last level (children are leaves)."""
    if _is_leaf(x):
        return False
    if isinstance(x, dict):
        vs = x.values()
    else:
        vs = x
    return all(map(_is_leaf, vs))


def _is_last_level_nonempty(x):
    """Returns whether pytree is at the last level and has any child."""
    return _is_last_level(x) and x


def nested_unzip(x):
    """Uzips a pytree of lists.

    Inverse of nested_zip.

    Example:
        nested_unzip((([1, 4], [2, 5]), [3, 6])) == [((1, 2), 3), ((4, 5), 6)]
    """
    if not x:
        raise ValueError(f'Cannot infer expected output structure from empty '
                         f'object (got {x} of type {type(x)})')
    acc = []
    try:
        i = 0
        while True:
            acc.append(nested_map(
                lambda l: l[i],
                x,
                stop_fn=_is_last_level_nonempty,
            ))
            i += 1
    except IndexError:
        return acc


@_verbose(arg_index=1)
def nested_zip_with(f, xs):
    """Zips a list of pytrees using a function.

    Example (elementwise addition of 2 pytrees):
        nested_zip_with(
            lambda x, y: x + y, [((1, 2), 3), ((4, 5), 6)]
        ) == ((5, 7), 9)
    """
    _assert_zippable(xs)

    def f_star(args):
        return f(*args)
    return nested_map(
        f_star, nested_zip.__wrapped__(xs), stop_fn=_is_last_level_nonempty
    )


def nested_stack(xs):
    """Stacks a list of pytrees of numpy arrays.

    Inverse of nested_unstack.

    Example:
        nested_stack([(1, 2), (3, 4)]) == (np.array([1, 3]), np.array([2, 4]))
    """
    return nested_map(np.stack, nested_zip(xs), stop_fn=_is_last_level_nonempty)


def nested_unstack(x):
    """Unstacks a pytree of numpy arrays.

    Inverse of nested_unstack.

    Example:
        nested_unstack((np.array([1, 3]), np.array([2, 4]))) == [(1, 2), (3, 4)]
    """
    def unstack(arr):
        (*slices,) = arr
        return slices
    return nested_unzip(nested_map(unstack, x))


def nested_concatenate(xs):
    """Concatenates a list of pytrees of numpy arrays."""
    return nested_map(
        np.concatenate, nested_zip(xs), stop_fn=_is_last_level_nonempty
    )


@_verbose(arg_index=1)
def nested_reduce(f, x, to_list=True, stop_fn=_is_leaf):
    """Reduces a pytree to a single value.

    Example:
        nested_reduce(sum, ((1, 2), 3)) == 6

    Args:
        f (callable): The reducer - function collection[T] -> T, where
            collection is list if to_list is True, or
            list/tuple/namedtuple/dict otherwise.
        x: pytree[T] to reduce.
        to_list (bool): Whether to convert collections to lists before passing
            them to the reducer.
        stop_fn (callable): Function pytree[T] -> bool determining when to stop
            the recursion.

    Returns:
        T: The reduced value.
    """
    if stop_fn(x):
        return x

    recurse = functools.partial(
        nested_reduce.__wrapped__, to_list=to_list, stop_fn=stop_fn
    )

    if isinstance(x, dict):
        collection = {
            k: recurse(f, v) for (k, v) in x.items()
        }
        if to_list:
            collection = list(collection.values())
    elif _is_namedtuple_instance(x):
        return recurse(f, tuple(x))
    else:
        assert isinstance(x, (list, tuple)), (
            'Non-exhaustive pattern match for {}.'.format(type(x))
        )
        collection = type(x)(recurse(f, y) for y in x)
    if to_list:
        collection = list(collection)

    return f(collection)


def nested_array_equal(x1, x2):
    """Tests the equality of two pytrees of numpy arrays."""
    return nested_reduce(all, nested_zip_with(np.array_equal, (x1, x2)))


def broadcast_to_pytree_shape(value, pytree):
    """Creates a new pytree of the same shape filled with the given value."""
    return nested_map(lambda _: value, pytree)


def choose_leaf(x):
    """Chooses a leaf of the pytree."""
    class Found(Exception):
        pass

    def find_leaf(x):
        raise Found(x)

    try:
        nested_map(find_leaf, x)
        raise ValueError('Pytree has no leaves.')
    except Found as e:
        (leaf,) = e.args
        return leaf


def full_pytree(signature, fill_value, shape_prefix=()):
    """Builds a constant-filled pytree of a given signature.

    Args:
        signature (pytree): Pytree of TensorSignature.
        fill_value: Value to fill each array with.
        shape_prefix (tuple): Shape to be prepended to each constructed array's
            shape.

    Returns:
        Pytree of a given signature with zero arrays as leaves.
    """
    return nested_map(
        lambda sig: np.full(
            shape=(shape_prefix + sig.shape),
            fill_value=fill_value,
            dtype=sig.dtype,
        ),
        signature,
    )


def zero_pytree(signature, shape_prefix=()):
    """Builds a zero-filled pytree of a given signature."""
    return full_pytree(signature, 0, shape_prefix)


def one_pytree(signature, shape_prefix=()):
    """Builds a one-filled pytree of a given signature."""
    return full_pytree(signature, 1, shape_prefix)
