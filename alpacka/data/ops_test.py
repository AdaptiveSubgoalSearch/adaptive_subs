"""Tests for alpacka.data.ops."""

import collections
import copy

import numpy as np
import pytest

from alpacka.data import ops


_TestNamedtuple = collections.namedtuple('_TestNamedtuple', ['test_field'])


def test_nested_map():
    inp = [{'x': (1, 2, 3), 'y': 4}, _TestNamedtuple(5)]
    out = ops.nested_map(lambda x: x + 1, inp)
    assert out == [{'x': (2, 3, 4), 'y': 5}, _TestNamedtuple(6)]


def test_nested_zip():
    inp = [_TestNamedtuple({'x': 1, 'y': 2}), _TestNamedtuple({'x': 3, 'y': 4})]
    out = ops.nested_zip(inp)
    assert out == _TestNamedtuple({'x': [1, 3], 'y': [2, 4]})


def test_nested_unzip():
    inp = _TestNamedtuple([1, 2])
    out = ops.nested_unzip(inp)
    assert out == [_TestNamedtuple(1), _TestNamedtuple(2)]


def test_nested_zip_with():
    inp = [((1, 2), 3), ((4, 5), 6)]
    out = ops.nested_zip_with(lambda x, y: x + y, inp)
    assert out == ((5, 7), 9)


def test_nested_stack_unstack():
    inp = [_TestNamedtuple(1), _TestNamedtuple(2)]
    out = ops.nested_unstack(ops.nested_stack(inp))
    assert inp == out


def test_nested_unstack_stack():
    inp = _TestNamedtuple(np.array([1, 2]))
    out = ops.nested_unstack(ops.nested_stack(inp))
    np.testing.assert_equal(inp, out)


@pytest.mark.parametrize('empty', [(), [], {}])
def test_nested_stack_empty(empty):
    inp = [empty, empty]
    out = ops.nested_stack(inp)
    assert out == empty


@pytest.mark.parametrize('empty', [(), [], {}])
def test_nested_concatenate_empty(empty):
    inp = [empty, empty]
    out = ops.nested_concatenate(inp)
    assert out == empty


def test_nested_concatenate():
    inp = (
        _TestNamedtuple(np.array([1, 2])),
        _TestNamedtuple(np.array([3])),
    )
    out = ops.nested_concatenate(inp)
    np.testing.assert_equal(out, _TestNamedtuple(np.array([1, 2, 3])))


def test_nested_reduce_sum():
    inp = [{'x': (1, 2, 3), 'y': 4}, _TestNamedtuple(5)]
    assert ops.nested_reduce(sum, inp) == 15


def test_nested_reduce_flatten_dict():
    """Flattens a nested dict into a dict indexed by paths in the tree."""
    inp = {'a': {'b': 1, 'c': 2}, 'd': 3}

    def flatten_dict(outer_dict):
        def flatten_inner(outer_key, inner_dict):
            return {
                f'{outer_key}/{inner_key}': value
                for (inner_key, value) in inner_dict.items()
            }

        flat_dict = {}
        for (outer_key, outer_value) in outer_dict.items():
            if isinstance(outer_value, dict):
                flat_dict.update(flatten_inner(outer_key, outer_value))
            else:
                flat_dict[outer_key] = outer_value
        return flat_dict

    out = ops.nested_reduce(flatten_dict, inp, to_list=False)
    assert out == {'a/b': 1, 'a/c': 2, 'd': 3}


def test_nested_array_equal():
    inp = (
        _TestNamedtuple(np.array([1, 2])),
        _TestNamedtuple(np.array([3])),
    )
    assert ops.nested_array_equal(inp, copy.deepcopy(inp))


def test_nested_array_not_equal():
    inp1 = (
        _TestNamedtuple(np.array([1, 2])),
        _TestNamedtuple(np.array([3])),
    )
    inp2 = (
        _TestNamedtuple(np.array([1, 4])),
        _TestNamedtuple(np.array([3])),
    )
    assert not ops.nested_array_equal(inp1, inp2)


def test_choose_leaf():
    inp = _TestNamedtuple(([123, 123], 123))
    out = ops.choose_leaf(inp)
    assert out == 123


def test_choose_leaf_raises_for_no_leaves():
    inp = (_TestNamedtuple([(), ()]),)
    with pytest.raises(ValueError):
        ops.choose_leaf(inp)
