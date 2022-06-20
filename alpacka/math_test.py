"""Tests for alpacka.math."""

import flaky
import numpy as np
import pytest

from alpacka import math
from alpacka import testing


def random_with_rank(rank):
    shape = range(1, rank + 1)
    return np.random.random_sample(shape)


@pytest.fixture(params=[1, 3])
def n_categories(request):
    return request.param


@pytest.fixture(params=[2, 3])
def n_categories_plural(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def rank(request):
    return request.param


@pytest.fixture(params=[False, True])
def keep_last_dim(request):
    return request.param


def test_log_sum_exp_shape(rank, keep_last_dim):
    x = random_with_rank(rank)
    y = math.log_sum_exp(x, keep_last_dim=keep_last_dim)
    expected_shape = x.shape[:-1]
    if keep_last_dim:
        expected_shape += (1,)
    assert y.shape == expected_shape


def test_log_sum_exp_inequalities(n_categories):
    x = np.random.random_sample(n_categories)
    y = math.log_sum_exp(x)
    testing.assert_array_less_or_close(np.max(x), y)
    testing.assert_array_less_or_close(y, np.max(x) + np.log(n_categories))


def test_log_mean_exp_shape(rank, keep_last_dim):
    x = random_with_rank(rank)
    y = math.log_mean_exp(x, keep_last_dim=keep_last_dim)
    expected_shape = x.shape[:-1]
    if keep_last_dim:
        expected_shape += (1,)
    assert y.shape == expected_shape


def test_log_mean_exp_inequalities(n_categories):
    x = np.random.random_sample(n_categories)
    y = math.log_mean_exp(x)
    testing.assert_array_less_or_close(np.max(x) - np.log(n_categories), y)
    testing.assert_array_less_or_close(y, np.max(x))


def test_softmax_output_is_positive(n_categories):
    logits = np.random.random_sample(n_categories)
    np.testing.assert_array_less(0, math.softmax(logits))


def test_softmax_output_sums_to_one(n_categories):
    logits = np.random.random_sample(n_categories)
    np.testing.assert_almost_equal(np.sum(math.softmax(logits)), 1)


def test_softmax_shape(rank):
    x = random_with_rank(rank)
    y = math.softmax(x)
    assert x.shape == y.shape


def test_log_softmax_equals_log_of_softmax(rank):
    x = random_with_rank(rank)
    log_softmax = math.log_softmax(x)
    softmax = math.softmax(x)
    np.testing.assert_almost_equal(log_softmax, np.log(softmax))


def test_categorical_entropy_uniform_logits(n_categories):
    logits = np.ones(n_categories)
    entropy = math.categorical_entropy(logits=logits)
    np.testing.assert_almost_equal(entropy, np.log(n_categories))


def test_categorical_entropy_uniform_probs(n_categories):
    probs = np.ones(n_categories) / n_categories
    entropy = math.categorical_entropy(probs=probs)
    np.testing.assert_almost_equal(entropy, np.log(n_categories))


def test_categorical_entropy_deterministic_probs(n_categories):
    probs = np.zeros(n_categories)
    probs[np.random.randint(n_categories)] = 1
    entropy = math.categorical_entropy(probs=probs)
    np.testing.assert_almost_equal(entropy, 0)


@pytest.mark.parametrize(
    'fn', [math.categorical_entropy, math.categorical_sample]
)
def test_raises_when_some_probs_are_negative(fn, n_categories_plural):
    # Exactly one entry is negative and all sum up to 1.
    probs = np.ones(n_categories_plural)
    probs[np.random.randint(n_categories_plural)] = -0.5
    probs /= np.sum(probs)

    with pytest.raises(ValueError):
        fn(probs=probs)


@pytest.mark.parametrize(
    'fn', [math.categorical_entropy, math.categorical_sample]
)
def test_raises_when_probs_dont_sum_to_one(fn, n_categories):
    probs = np.ones(n_categories) / (n_categories + 1)
    with pytest.raises(ValueError):
        fn(probs=probs)


def test_categorical_entropy_shape_logits(rank):
    x = random_with_rank(rank)

    y = math.categorical_entropy(logits=x, mean=True)
    assert np.isscalar(y)

    y = math.categorical_entropy(logits=x, mean=False)
    assert y.shape == x.shape[:-1]


def test_categorical_entropy_shape_probs(rank):
    x = math.softmax(random_with_rank(rank))

    y = math.categorical_entropy(probs=x, mean=True)
    assert np.isscalar(y)

    y = math.categorical_entropy(probs=x, mean=False)
    assert y.shape == x.shape[:-1]


def monte_carlo_sample(n, logits=None, probs=None):
    n_categories = len(logits) if logits is not None else len(probs)
    histogram = np.zeros(n_categories)
    for _ in range(n):
        sample = math.categorical_sample(logits=logits, probs=probs)
        histogram[sample] += 1
    return histogram / np.sum(histogram)


@flaky.flaky
def test_categorical_sample_law_of_big_numbers_probs(n_categories):
    probs = np.random.random_sample(n_categories)
    probs /= np.sum(probs)
    histogram = monte_carlo_sample(probs=probs, n=1000)
    np.testing.assert_allclose(histogram, probs, rtol=0.5)


@flaky.flaky
def test_categorical_sample_law_of_big_numbers_logits(n_categories):
    logits = np.random.random_sample(n_categories)
    histogram = monte_carlo_sample(logits=logits, n=1000)
    np.testing.assert_allclose(histogram, math.softmax(logits), rtol=0.5)


def test_categorical_sample_shape_logits(rank):
    x = random_with_rank(rank)
    y = math.categorical_sample(logits=x)
    assert y.shape == x.shape[:-1]


def test_categorical_sample_shape_probs(rank):
    x = math.softmax(random_with_rank(rank))
    y = math.categorical_sample(probs=x)
    assert y.shape == x.shape[:-1]
