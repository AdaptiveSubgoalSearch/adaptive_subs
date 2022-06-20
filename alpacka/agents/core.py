"""Core agents."""

import numpy as np

from alpacka import data
from alpacka.agents import base
from alpacka.utils import metric as metric_utils
from alpacka.utils import space as space_utils


class ActorCriticAgent(base.OnlineAgent):
    """Agent that uses value and policy networks to infer value and logits."""

    def __init__(self, distribution, **kwargs):
        """Initializes ActorCriticAgent.

        Args:
            distribution (ProbabilityDistribution): Probability distribution
                parameterized by the inferred logits to sample actions from and
                calculate statistics put into an agent info.
            kwargs: OnlineAgent init keyword arguments.
        """
        super().__init__(**kwargs)
        self.distribution = distribution

    def act(self, observation):
        batched_value, batched_logits = yield np.expand_dims(observation,
                                                             axis=0)
        value = np.squeeze(batched_value, axis=0)  # Removes batch dim.
        logits = np.squeeze(batched_logits, axis=0)  # Removes batch dim.

        action = self.distribution.sample(logits)

        agent_info = {'value': value,
                      'logits': logits}
        agent_info.update(self.distribution.compute_statistics(logits))

        return action, agent_info

    def network_signature(self, observation_space, action_space):
        return data.NetworkSignature(
            input=space_utils.signature(observation_space),
            output=(data.TensorSignature(shape=(1,)),
                    self.distribution.params_signature(action_space))
        )

    @staticmethod
    def compute_metrics(episodes):
        agent_info_batch = data.nested_concatenate(
            [episode.transition_batch.agent_info for episode in episodes])
        metrics = {}

        metrics.update(metric_utils.compute_scalar_statistics(
            agent_info_batch['value'],
            prefix='value',
            with_min_and_max=True
        ))
        metrics.update(metric_utils.compute_scalar_statistics(
            agent_info_batch['logits'],
            prefix='logits',
            with_min_and_max=True
        ))

        return metrics


class PolicyNetworkAgent(base.OnlineAgent):
    """Agent that uses a policy network to infer logits."""

    def __init__(self, distribution, **kwargs):
        """Initializes PolicyNetworkAgent.

        Args:
            distribution (ProbabilityDistribution): Probability distribution
                parameterized by the inferred logits to sample actions from and
                calculate statistics put into an agent info.
            kwargs: OnlineAgent init keyword arguments.
        """
        super().__init__(**kwargs)
        self.distribution = distribution

    def act(self, observation):
        batched_logits = yield np.expand_dims(observation, axis=0)
        logits = np.squeeze(batched_logits, axis=0)  # Removes batch dim.

        action = self.distribution.sample(logits)

        agent_info = {'logits': logits}
        agent_info.update(self.distribution.compute_statistics(logits))

        return action, agent_info

    def network_signature(self, observation_space, action_space):
        return data.NetworkSignature(
            input=space_utils.signature(observation_space),
            output=self.distribution.params_signature(action_space),
        )

    @staticmethod
    def compute_metrics(episodes):
        agent_info_batch = data.nested_concatenate(
            [episode.transition_batch.agent_info for episode in episodes])

        return metric_utils.compute_scalar_statistics(
            agent_info_batch['logits'],
            prefix='logits',
            with_min_and_max=True
        )


class RandomAgent(base.OnlineAgent):
    """Random agent, sampling actions from the uniform distribution."""

    def act(self, observation):
        del observation
        return (self._action_space.sample(), {})

        # To indicate it's a coroutine:
        yield  # pylint: disable=unreachable
