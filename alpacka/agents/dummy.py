"""Dummy agent for development of UnionNetwork.
"""

import random

import gin
import numpy as np

from alpacka import data
from alpacka.agents import base
from alpacka.utils import space
from alpacka.utils import transformations


@gin.configurable
class DummyTwoNetworkAgent(base.OnlineAgent):
    """Dummy agent for development of trainable models.

    It queries two separate networks - possibly in random order (configurable).
    """
    def __init__(self, random_order=False, **kwargs):
        super().__init__(**kwargs)
        self._random_order = random_order

    def act(self, observation):
        """Queries two networks and chooses some arbitary action.

        The method doesn't do anything meaningful. It just tries out
        request-response flow in the multiple-networks setup.
        """
        agent_request = data.AgentRequest(observation[np.newaxis, :])

        n_actions = space.max_size(self._action_space)
        action_to_query = random.randrange(0, n_actions)

        model_request = data.ModelRequest({
            'observation': observation[np.newaxis, :],
            'action': transformations.one_hot_encode(
                [action_to_query], n_actions
            )
        })

        if not self._random_order or random.randrange(0, 2) == 0:
            agent_response = yield agent_request
            model_response = yield model_request
        else:
            model_response = yield model_request
            agent_response = yield agent_request

        assert agent_response.shape == (1, 1)
        assert data.ops.nested_map(lambda arr: arr.shape, model_response) == {
            'next_observation': (1,) + observation.shape,
            'reward': (1, 1),
            'done': (1, 1),
        }

        value = agent_response.item()
        meaningless_sum = (
                value
                + np.sum(model_response['next_observation'])
                + model_response['reward'].item()
                + model_response['done'].item()
        )
        action = int(meaningless_sum * 1e9) % n_actions

        return action, {'value': value}

    def network_signature(self, observation_space, action_space):
        return {
            data.AgentRequest: data.NetworkSignature(
                input=space.signature(observation_space),
                output=data.TensorSignature(shape=(1,)),
            ),
            data.ModelRequest: data.NetworkSignature(
                input={
                    'observation': space.signature(observation_space),
                    'action': data.TensorSignature(
                        shape=(space.max_size(action_space),)
                    ),
                },
                output={
                    'next_observation': space.signature(observation_space),
                    'reward': data.TensorSignature(shape=(1,)),
                    'done': data.TensorSignature(shape=(1,)),
                },
            )
        }
