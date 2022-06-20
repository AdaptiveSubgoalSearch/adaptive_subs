"""Distributions used by agents to sample actions."""

import numpy as np

from alpacka import data
from alpacka import math
from alpacka.utils import space as space_utils


class ProbabilityDistribution:
    """Base class for probability distributions."""

    def compute_statistics(self, params):
        """Computes probabilities, log probabilities and distribution entropy.

        Args:
            params (np.ndarray): Distribution parameters.

        Returns:
            dict: with keys 'prob', 'logp', 'entropy'.
        """
        raise NotImplementedError()

    def sample(self, params):
        """Samples from the distribution.

        Args:
            params (np.ndarray): Distribution parameters.

        Returns:
            Distribution-dependent: sample from the distribution.
        """
        raise NotImplementedError()

    def params_signature(self, action_space):  # pylint: disable=redundant-returns-doc,useless-return
        """Defines the signature of parameters this dist. is parameterized by.

        Overriding is optional.

        Args:
            action_space (gym.Space): Environment action space.

        Returns:
            TensorSignature or None: Either the parameters tensor signature or
            None if the distribution isn't parameterized.
        """
        del action_space
        return None


class CategoricalDistribution(ProbabilityDistribution):
    """Categorical probabilistic distribution.

    Softmax with temperature.
    """

    def __init__(self, temperature, from_logits):
        """Initializes CategoricalDistribution.

        Args:
            temperature (float): Softmax temperature parameter.
            from_logits (bool): Whether the distribution is parametrized by
                logits. If not, it's parametrized by probabilities.
        """
        super().__init__()
        self.temperature = temperature
        self._from_logits = from_logits

    def _compute_logits(self, params):
        if self._from_logits:
            return params
        else:
            return np.log(params)

    def compute_statistics(self, params):
        """Computes softmax, log softmax and entropy with temperature."""
        logits = self._compute_logits(params)
        w_logits = logits / self.temperature

        logp = math.log_softmax(w_logits)
        prob = math.softmax(w_logits)
        entropy = math.categorical_entropy(logits=w_logits, mean=False)

        return {'prob': prob, 'logp': logp, 'entropy': entropy}

    def sample(self, params):
        """Sample from categorical distribution with temperature."""
        logits = self._compute_logits(params)
        w_logits = logits / self.temperature
        return math.categorical_sample(logits=w_logits)

    @staticmethod
    def params_signature(action_space):
        return data.TensorSignature(
            shape=(space_utils.max_size(action_space),)
        )


class EpsilonGreedyDistribution(ProbabilityDistribution):
    """Epsilon-greedy probability distribution."""

    def __init__(self, epsilon):
        """Initializes EpsilonGreedyDistribution.

        Args:
            epsilon (float): Probability of taking random action.
        """
        super().__init__()
        self.epsilon = epsilon

    def compute_statistics(self, params):
        prob = np.full(shape=params.shape,
                       fill_value=self.epsilon/len(params))
        prob[np.argmax(params)] += 1 - self.epsilon

        logp = np.log(prob + 1e-9)
        entropy = math.categorical_entropy(probs=prob)

        return {'prob': prob, 'logp': logp, 'entropy': entropy}

    def sample(self, params):
        if np.random.random_sample() < self.epsilon:
            return np.random.choice(len(params))
        else:
            return np.argmax(params)

    @staticmethod
    def params_signature(action_space):
        return data.TensorSignature(
            shape=(space_utils.max_size(action_space),)
        )
