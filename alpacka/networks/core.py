"""Deep learning framework-agnostic interface for neural networks."""

import os

import gin

from alpacka import data
from alpacka.utils import transformations


class Network:
    """Base class for networks."""

    def __init__(self, network_signature):
        """Initializes Network.

        Args:
            network_signature (NetworkSignature): Network signature.
        """
        self._network_signature = network_signature

    def clone(self):
        new_network = type(self)(network_signature=self._network_signature)
        new_network.params = self.params
        return new_network

    def predict(self, inputs):
        """Returns the prediction for a given input.

        Args:
            inputs (Agent-dependent): Batch of inputs to run prediction on.
        """
        raise NotImplementedError

    @property
    def params(self):
        """Returns network parameters."""
        raise NotImplementedError

    @params.setter
    def params(self, new_params):
        """Sets network parameters."""
        raise NotImplementedError

    def save(self, checkpoint_path):
        """Saves network parameters to a file."""
        raise NotImplementedError

    def restore(self, checkpoint_path):
        """Restores network parameters from a file."""
        raise NotImplementedError


class TrainableNetwork(Network):
    """Base class for networks that can be trained."""

    def train(self, data_stream, n_steps):
        """Performs one epoch of training on data prepared by the Trainer.

        Args:
            data_stream: (iterable) Python generator of triples
                (input, target, weight) to run the updates on.
            n_steps: (int) Number of training steps in the epoch.

        Returns:
            dict: Collected metrics, indexed by name.
        """
        raise NotImplementedError


class DummyNetwork(TrainableNetwork):
    """Dummy TrainableNetwork for testing."""

    def __init__(self, network_signature):
        super().__init__(network_signature)
        self._params = ''

    def train(self, data_stream, n_steps):
        del data_stream
        return {}

    def predict(self, inputs):
        if self._network_signature is None:
            # If no network signature has been specified, just return whatever -
            # for instance, the inputs.
            return inputs
        else:
            # Check if the input has correct shapes. Ignore the batch dimension.
            # We don't check the dtype because some envs don't respect what they
            # defined in their observation_space and it causes problems.
            input_shapes = data.nested_map(lambda x: x.shape[1:], inputs)
            expected_shapes = data.nested_map(
                lambda x: x.shape, self._network_signature.input
            )
            assert input_shapes == expected_shapes, (
                f'Incorrect input shapes: {input_shapes} != {expected_shapes}.'
            )

            # Return a dummy pytree of the required signature, with the same
            # batch dimension as the input.
            batch_size = data.choose_leaf(inputs).shape[0]  # pylint: disable=no-member
            return data.zero_pytree(
                self._network_signature.output, shape_prefix=(batch_size,)
            )

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        self._params = new_params

    def save(self, checkpoint_path):
        with open(checkpoint_path, 'w') as f:
            f.write(self._params)

    def restore(self, checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            self._params = f.read()


class UnionNetwork(Network):
    """Network consisting of multiple networks.

    Useful for the case, when the agent needs to use more than one network.

    Please note that UnionNetwork doesn't comply with the TrainableNetwork
    interface. For training you need to use the dedicated UnionTrainer.
    """

    def __init__(self, network_signature, request_to_network=gin.REQUIRED):
        """Initializes UnionNetwork.

        Args:
            network_signature (dict(type -> NetworkSignature)):
                For every request type specifies the signature of a network
                to handle requests of that type. Key: type of request,
                value: NetworkSignature.
            request_to_network (dict(type -> callable)):
                For every request type specifies a separate network to handle
                requests of that type. Key: type of request, value: function to
                create a network.
        """
        super().__init__(network_signature)

        [networks, signatures] = [
            transformations.map_dict_keys(dictionary, data.request_type_id)
            for dictionary in [request_to_network, network_signature]
        ]

        self._networks = {
            type_id: network_fn(sig)
            for type_id, (network_fn, sig) in transformations.zip_dicts_strict(
                networks, signatures
            ).items()
        }

    def predict(self, inputs):
        type_id = data.request_type_id(type(inputs))
        network = self._networks[type_id]
        return network.predict(inputs.value)

    @property
    def subnetworks(self):
        return self._networks

    @property
    def params(self):
        return {
            type_id: network.params
            for type_id, network in self._networks.items()
        }

    @params.setter
    def params(self, new_params):
        if new_params.keys() != self._networks.keys():
            raise ValueError(
                'Keys of new_params do not match stored networks.\n'
                f'{new_params.keys()} != {self._networks.keys()}'
            )

        for type_id, network in self._networks.items():
            network.params = new_params[type_id]

    def save(self, checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)
        for (slug, network) in self.subnetworks.items():
            network.save(os.path.join(checkpoint_path, slug))

    def restore(self, checkpoint_path):
        for (slug, network) in self.subnetworks.items():
            network.restore(os.path.join(checkpoint_path, slug))
