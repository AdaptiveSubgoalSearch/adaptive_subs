"""Test data transformation utils."""

import numpy as np
import pytest

from alpacka.utils import transformations


def test_one_hot_encode_basic_usage():
    values = [2, 0, -1, 1]
    space_size = 3
    expected = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ])
    assert np.array_equal(
        expected, transformations.one_hot_encode(values, space_size)
    )
    assert np.array_equal(
        expected, transformations.one_hot_encode(np.array(values), space_size)
    )


def test_one_hot_encode_edge_cases():
    assert np.array_equal(
        transformations.one_hot_encode([1], 2),
        np.array([[0, 1]])
    )
    assert np.array_equal(
        transformations.one_hot_encode([], 2),
        np.empty(shape=(0, 2))
    )


def test_one_hot_encode_dtype():
    values = [0, 1, 2]

    assert transformations.one_hot_encode(
        values, 3, dtype=np.float32
    ).dtype == np.float32

    assert transformations.one_hot_encode(
        values, 3, dtype=np.int8
    ).dtype == np.int8


def test_one_hot_encode_invalid_args():
    # Value out of bounds.
    with pytest.raises(IndexError):
        transformations.one_hot_encode(
            [2], 2
        )
    with pytest.raises(IndexError):
        transformations.one_hot_encode(
            [-3], 2
        )

    # Negative action_space_size.
    with pytest.raises(ValueError):
        transformations.one_hot_encode(
            [], -1
        )

    # Non-integer values.
    with pytest.raises(IndexError):
        transformations.one_hot_encode(
            [1.0], 3
        )
    with pytest.raises(IndexError):
        transformations.one_hot_encode(
            np.array([1], dtype=np.float32), 3
        )


def test_map_dict_keys_basic_usage():
    assert transformations.map_dict_keys(
        {1: 2, 3: 4}, lambda x: 2 * x
    ) == {2: 2, 6: 4}

    assert transformations.map_dict_keys({}, lambda x: x) == {}


def test_map_dict_keys_collision():
    with pytest.raises(ValueError):
        transformations.map_dict_keys(
            {'key': 1, 'KEY': 10}, lambda s: s.lower()
        )


def test_zip_dict_strict_basic_usage():
    assert transformations.zip_dicts_strict(
        {'key1': 1, 'key2': 2},
        {'key1': 3, 'key2': 4}
    ) == {
        'key1': (1, 3), 'key2': (2, 4),
    }


def test_zip_dict_strict_key_mismatch():
    subdict = {
        'key1': 1, 'key2': 2
    }
    superdict = subdict.copy()
    superdict['key3'] = 3

    with pytest.raises(ValueError):
        transformations.zip_dicts_strict(subdict, superdict)
    with pytest.raises(ValueError):
        transformations.zip_dicts_strict(superdict, subdict)
    with pytest.raises(ValueError):
        transformations.zip_dicts_strict(subdict, {'key1': 1, 'key_other': 4})
