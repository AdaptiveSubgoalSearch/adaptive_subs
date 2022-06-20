"""Datatypes and functions for manipulating them."""

import collections

import gin

from alpacka.data.ops import *


# Transition between two states, S and S'.
Transition = collections.namedtuple(
    'Transition',
    [
        # Observation obtained at S.
        'observation',
        # Action played based on the observation.
        'action_list',
        # Reward obtained after performing the action.
        'reward',
        # Whether the environment is "done" at S'.
        'done',
        # Observation obtained at S'.
        'next_observation',
        # Dict of any additional info supplied by the agent.
        'agent_info',
    ]
)


# Basic Episode object, summarizing experience collected when solving an episode
# in the form of transitions. It's used for basic Agent -> Trainer
# communication. Agents and Trainers can use different (but shared) episode
# representations, as long as they have a 'return_' field, as this field is used
# by Runner for reporting metrics.
Episode = collections.namedtuple(
    'Episode',
    [
        # Transition object containing a batch of transitions.
        'transitions',
        # Undiscounted return (cumulative reward) for the entire episode.
        'return_',
        # Whether the episode was "solved".
        'solved',
        # # Whether the episode was truncated by the TimeLimitWrapper.
        # 'truncated',
        # # Size of the environment's action space.
        # 'action_space_size',
    ]
)
# # Defaults for solved, truncated and action_space_size.
# Episode.__new__.__defaults__ = (None, None, None)


# Signature of a tensor. Contains shape and datatype - the static information
# needed to initialize a tensor, for example a numpy array.
TensorSignature = collections.namedtuple(
    'TensorSignature', ['shape', 'dtype']
)
TensorSignature.__new__.__defaults__ = (np.float32,)  # dtype
# Register TensorSignature as a leaf type, so we can for example do nested_map
# over a structure of TensorSignatures to initialize a pytree of arrays.
register_leaf_type(TensorSignature)


# Signature of a network: input -> output. Both input and output are pytrees of
# TensorSignatures.
NetworkSignature = collections.namedtuple(
    'NetworkSignature', ['input', 'output']
)


# Classes and functions related to querying networks

# Please note that the following code addresses just some rare use cases. In
# most cases you don't need any special datatypes for querying a network - you
# can simply use numpy arrays.

# See README.md for more info about usual requests.
# See alpacka.batch_steppers.core.RequestHandler for NetworkRequest.
# See alpacka.networks.core.UnionNetwork for named prediction requests.

# Request of a network_fn: Function () -> Network and the current parameters.
class NetworkRequest:
    pass


def request_type_id(request_type):
    """Returns type id, which is comparable and hashable.

    It is useful for matching Agent's requests by type. request_type has to be
    registered using register_prediction_request() function beforehand.

    We don't compare requests just by type(), as it doesn't work well with Gin
    and Ray.
    """
    try:
        return request_type.slug
    except AttributeError:
        raise ValueError(
            'Expected request type registered by '
            'alpacka.data.register_prediction_request() function.'
        )


def register_prediction_request(name, slug=None, module='alpacka.data'):
    """Creates new request type and registers it in gin.

    UnionNetwork checks request type to determine, which Network should
    process that request. This function is useful, if you want to create your
    own type of request for your agent.

    For single Networks, you'd be better off using simple np.arrays instead.

    Argument 'slug' equals to 'name' argument by default. 'slug' should be
    unique for every invocation of this function.
    """
    request_type = collections.namedtuple(name, ['value'], module=module)
    request_type.slug = slug or name

    gin_type = gin.external_configurable(request_type, module=module)

    type_id = request_type_id(gin_type)
    if type_id in register_prediction_request.taken_ids:
        raise ValueError(f'Request id {type_id} is already taken.')
    register_prediction_request.taken_ids.add(type_id)

    return gin_type


register_prediction_request.taken_ids = set()


AgentRequest = register_prediction_request('AgentRequest', slug='agent')  # pylint: disable=invalid-name

ModelRequest = register_prediction_request('ModelRequest', slug='model')  # pylint: disable=invalid-name
