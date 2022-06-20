"""Uniform replay buffer."""

import functools

import numpy as np
import randomdict

from alpacka import data


class UniformReplayBuffer:
    """Replay buffer with uniform sampling.

    Stores datapoints in a queue of fixed size. Adding to a full buffer
    overwrites the oldest ones.
    """

    def __init__(self, datapoint_signature, capacity):
        """Initializes the replay buffer.

        Args:
            datapoint_signature (pytree): Pytree of TensorSignatures, defining
                the structure of data to be stored.
            capacity (int): Maximum size of the buffer.
        """
        self._datapoint_shape = data.nested_map(
            lambda x: x.shape, datapoint_signature
        )
        self._capacity = capacity
        self._size = 0
        self._insert_index = 0

        def init_array(signature):
            shape = (self._capacity,) + signature.shape
            return np.zeros(shape, dtype=signature.dtype)
        self._data_buffer = data.nested_map(init_array, datapoint_signature)

    def add(self, stacked_datapoints):
        """Adds datapoints to the buffer.

        Args:
            stacked_datapoints (pytree): Transition object containing the
                datapoints, stacked along axis 0.

        Raises:
            ValueError: In case of a signature mismatch.
        """
        datapoint_shape = data.nested_map(
            lambda x: x.shape[1:], stacked_datapoints
        )
        if datapoint_shape != self._datapoint_shape:
            raise ValueError(
                'Datapoint shape mismatch: got {}, expected {}.'.format(
                    datapoint_shape, self._datapoint_shape
                )
            )

        n_elems = data.choose_leaf(data.nested_map(
            lambda x: x.shape[0], stacked_datapoints
        ))

        def insert_to_array(buf, elems):
            buf_size = buf.shape[0]
            assert elems.shape[0] == n_elems
            index = self._insert_index
            # Insert up to buf_size at the current index.
            buf[index:min(index + n_elems, buf_size)] = elems[:buf_size - index]
            # Insert whatever's left at the beginning of the buffer.
            buf[:max(index + n_elems - buf_size, 0)] = elems[buf_size - index:]

        # Insert to all arrays in the pytree.
        data.nested_zip_with(
            insert_to_array, (self._data_buffer, stacked_datapoints)
        )
        if self._size < self._capacity:
            self._size = min(self._insert_index + n_elems, self._capacity)
        self._insert_index = (self._insert_index + n_elems) % self._capacity

    def sample(self, batch_size):
        """Samples a batch of datapoints.

        Args:
            batch_size (int): Number of datapoints to sample.

        Returns:
            Datapoint object with sampled datapoints stacked along the 0 axis.

        Raises:
            ValueError: If the buffer is empty.
        """
        if self._data_buffer is None:
            raise ValueError('Cannot sample from an empty buffer.')
        indices = np.random.randint(self._size, size=batch_size)
        return data.nested_map(lambda x: x[indices], self._data_buffer)


class HierarchicalReplayBuffer:
    """Replay buffer with hierarchical sampling.

    Datapoints are indexed by a list of "buckets". Buckets are sampled
    uniformly, in a fixed order. Each sequence of buckets has its own capacity.
    """

    def __init__(self, datapoint_signature, capacity, hierarchy_depth):
        """Initializes HierarchicalReplayBuffer.

        Args:
            datapoint_signature (pytree): Pytree of TensorSignatures, defining
                the structure of data to be stored.
            capacity (int): Maximum size of the buffer.
            hierarchy_depth (int): Number of buckets in the hierarchy.
        """
        self._raw_buffer_fn = functools.partial(
            UniformReplayBuffer, datapoint_signature, capacity
        )
        # Data is stored in a tree, where inner nodes are dicts with fast
        # random sampling (RandomDicts) and leaves are UniformReplayBuffers.
        # This won't scale to buckets with a large number of possible values.
        if hierarchy_depth:
            self._buffer_hierarchy = randomdict.RandomDict()
        else:
            self._buffer_hierarchy = self._raw_buffer_fn()
        self._hierarchy_depth = hierarchy_depth

    def add(self, stacked_datapoints, buckets):
        """Adds datapoints to the buffer.

        Args:
            stacked_datapoints (pytree): Transition object containing the
                datapoints, stacked along axis 0.
            buckets (list): List of length hierarchy_depth with values of
                buckets.
        """
        assert len(buckets) == self._hierarchy_depth

        # Because Python doesn't support value assignment, we need to if out
        # the case when the hierarchy is flat.
        if self._hierarchy_depth:
            buffer_hierarchy = self._buffer_hierarchy
            for bucket in buckets[:-1]:
                if bucket not in buffer_hierarchy:
                    buffer_hierarchy[bucket] = randomdict.RandomDict()
                buffer_hierarchy = buffer_hierarchy[bucket]

            bucket = buckets[-1]
            if bucket not in buffer_hierarchy:
                buffer_hierarchy[bucket] = self._raw_buffer_fn()
            buf = buffer_hierarchy[bucket]
        else:
            buf = self._buffer_hierarchy

        buf.add(stacked_datapoints)

    def _sample_one(self):
        buffer_hierarchy = self._buffer_hierarchy
        for _ in range(self._hierarchy_depth):
            buffer_hierarchy = buffer_hierarchy.random_value()
        return buffer_hierarchy.sample(batch_size=1)

    def sample(self, batch_size):
        """Samples a batch of datapoints.

        Args:
            batch_size (int): Number of datapoints to sample.

        Returns:
            Datapoint object with sampled datapoints stacked along the 0 axis.

        Raises:
            ValueError: If the buffer is empty.
        """
        return data.nested_concatenate(
            [self._sample_one() for _ in range(batch_size)]
        )
