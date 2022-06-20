"""Math functions."""

import numpy as np


def log_sum_exp(logits, keep_last_dim=False):
    """Computes log(sum_i(exp(x_i))).

    Args:
        logits: Array of shape (..., n_categories).
        keep_last_dim (bool): Whether to keep the last dimension of the output.

    Returns:
        Array of shape (..., n_categories) with the computed log-sum-exp over
        the last dimension.
    """
    logits = np.array(logits)
    baseline = np.max(logits, axis=-1, keepdims=True)
    result = np.log(
        np.sum(np.exp(logits - baseline), axis=-1, keepdims=True)
    ) + baseline
    if not keep_last_dim:
        result = np.squeeze(result, axis=-1)
    return result


def log_mean_exp(logits, keep_last_dim=False):
    """Computes log(1/n sum_i(exp(x_i))).

    Args:
        logits: Array of shape (..., n_categories).
        keep_last_dim (bool): Whether to keep the last dimension of the output.

    Returns:
        Array of shape (..., n_categories) with the computed log-sum-exp over
        the last dimension.
    """
    logits = np.array(logits)
    return log_sum_exp(logits, keep_last_dim) - np.log(logits.shape[-1])


def log_softmax(logits):
    """Log-softmax.

    Args:
        logits: Array of shape (..., n_categories).

    Returns:
        Array of shape (..., n_categories) with the computed log-softmax over
        the last dimension.
    """
    return logits - log_sum_exp(logits, keep_last_dim=True)


def softmax(logits):
    """Softmax.

    Args:
        logits: Array of shape (..., n_categories).

    Returns:
        Array of shape (..., n_categories) with the computed softmax, summing
        up to 1 over the last dimension.
    """
    return np.exp(log_softmax(logits))


def _validate_categorical_params(logits, probs):
    if (logits is None) == (probs is None):
        raise ValueError(
            'Either logits or probs must be provided (exactly one has to be '
            'not None).'
        )

    if probs is not None:
        if np.any(probs < 0):
            raise ValueError('Some probabilities are negative.')

        if not np.allclose(np.sum(probs, axis=-1), 1):
            raise ValueError('Probabilities don\'t sum to one.')


def categorical_entropy(logits=None, probs=None, mean=True, epsilon=1e-9):
    """Entropy of a categorical distribution.

    Works either on logits or on probabilities.

    Args:
        logits: Array of shape (..., n_categories). Don't need to be normalized.
        probs: Array of shape (..., n_categories). Need to be normalized.
        mean (bool): Whether to compute a mean over the leading dimensions, or
            to leave them as they are.
        epsilon (float): Small constant for numerical stability.

    Returns:
        Scalar mean entropy if mean is True, or array of shape (...) of
        elementwise entropies otherwise.

    Raises:
        ValueError: If either none or both of logits and probs are provided, or
        if the probabilities don't sum to one.
    """
    _validate_categorical_params(logits, probs)

    if probs is not None:
        entropy = -np.sum(np.array(probs) * np.log(probs + epsilon), axis=-1)

    if logits is not None:
        logits = log_softmax(logits)
        entropy = -np.sum(np.exp(logits) * logits, axis=-1)

    if mean:
        entropy = np.mean(entropy)
    return entropy


def categorical_sample(logits=None, probs=None, epsilon=1e-9):
    """Categorical sampling using Gumbel noise.

    Works either on logits or on probabilities.

    Args:
        logits: Array of shape (..., n_categories). Don't need to be normalized.
        probs: Array of shape (..., n_categories). Need to be normalized.
        epsilon (float): Small constant for numerical stability.

    Returns:
        Array of shape (...) of samples.

    Raises:
        ValueError: If either none or both of logits and probs are provided, or
        if the probabilities don't sum to one.
    """
    _validate_categorical_params(logits, probs)

    if probs is not None:
        logits = np.log(np.array(probs) + epsilon)

    def gumbel_noise(shape):
        u = np.random.uniform(low=epsilon, high=(1.0 - epsilon), size=shape)
        return -np.log(-np.log(u))

    logits = np.array(logits)
    return np.argmax(logits + gumbel_noise(logits.shape), axis=-1)
