"""Supervised trainer."""

import time

import gin
import numpy as np

from alpacka import data
from alpacka.trainers import base
from alpacka.trainers import replay_buffers
from alpacka.utils import compression
from alpacka.utils import transformations


@gin.configurable
def input_observation(episode):
    return episode.transition_batch.observation


@gin.configurable
def input_action(episode):
    return transformations.one_hot_encode(
        episode.transition_batch.action,
        episode.action_space_size,
        dtype=np.float32
    )


@gin.configurable
def input_log_temperature(episode):
    return np.log(episode.transition_batch.agent_info['temperature'])[:, None]


@gin.configurable
def target_solved(episode):
    return np.full(
        shape=(episode.transition_batch.observation.shape[:1] + (1,)),
        fill_value=int(episode.solved),
    )


@gin.configurable
def target_return(episode):
    return np.cumsum(episode.transition_batch.reward[::-1],
                     dtype=np.float)[::-1, np.newaxis]


@gin.configurable
def target_discounted_return(episode):
    """Uses discounted_return calculated by agent."""
    return np.expand_dims(
        episode.transition_batch.agent_info['discounted_return'], axis=1
    )


@gin.configurable
def target_value(episode):
    return np.expand_dims(
        episode.transition_batch.agent_info['value'], axis=1
    )


@gin.configurable
def target_qualities(episode):
    return episode.transition_batch.agent_info['qualities']


@gin.configurable
def target_action_histogram(episode):
    return episode.transition_batch.agent_info['action_histogram']


@gin.configurable
def target_action_histogram_smooth(episode):
    return episode.transition_batch.agent_info['action_histogram_smooth']


@gin.configurable
def target_next_observation(episode):
    return episode.transition_batch.next_observation


@gin.configurable
def target_reward(episode):
    return np.expand_dims(episode.transition_batch.reward, axis=1)


@gin.configurable
def target_done(episode):
    return np.expand_dims(episode.transition_batch.done, axis=1)


@gin.configurable
def mask_one(episode, target):
    del episode
    return np.ones_like(target)


@gin.configurable
def mask_nonzero(episode, target):
    del episode
    return target != 0


@gin.configurable
def mask_action(episode, target):
    # The targets in this case should be provided separately for each action.
    n_actions = target.shape[-1]
    # One-hot vector, with 1 at the performed action.
    possible_actions = np.arange(n_actions)[None, :]
    episode_actions = episode.transition_batch.action[:, None]
    return possible_actions == episode_actions


class SupervisedTrainer(base.Trainer):
    """Supervised trainer.

    Trains the network based on (x, y) pairs generated out of transitions
    sampled from a replay buffer.
    """

    def __init__(
        self,
        network_signature,
        input=input_observation,  # pylint: disable=redefined-builtin
        target=target_solved,
        mask=None,
        batch_size=64,
        n_steps_per_epoch=1000,
        replay_buffer_capacity=1000000,
        replay_buffer_sampling_hierarchy=(),
    ):
        """Initializes SupervisedTrainer.

        Args:
            network_signature (pytree): Input signature for the network.
            input (pytree): Pytree of functions episode -> input for
                determining the inputs for network training. The structure of
                the tree should reflect the structure of an input.
            target (pytree): Counterpart to the input parameter, but for
                training targets.
            mask (pytree): Pytree of functions (episode, target) -> mask for
                determining the loss masks. The masks are specified
                per-element and must be of the same shape as targets.
            batch_size (int): Batch size.
            n_steps_per_epoch (int): Number of optimizer steps to do per
                epoch.
            replay_buffer_capacity (int): Maximum size of the replay buffer.
            replay_buffer_sampling_hierarchy (tuple): Sequence of Episode
                attribute names, defining the sampling hierarchy.
        """
        super().__init__(network_signature)

        # Go over the pytree of input/target functions to build a function
        # episode -> pytree of inputs/targets.
        def build_episode_to_pytree_mapper(functions_pytree):
            return lambda episode: data.nested_map(
                lambda f: f(episode), functions_pytree
            )

        self._input_fn = build_episode_to_pytree_mapper(input)
        self._target_fn = build_episode_to_pytree_mapper(target)

        if mask is None:
            mask = data.nested_map(lambda _: mask_one, target)
        # Zip the pytree of mask functions with the pytree of targets
        # to build a function episode -> pytree of masks.
        self._mask_fn = lambda episode: data.nested_zip_with(
            lambda f, target: f(episode, target),
            (mask, self._target_fn(episode))
        )

        self._batch_size = batch_size
        self._n_steps_per_epoch = n_steps_per_epoch

        # (input, target, mask)
        datapoint_sig = (
            network_signature.input,
            network_signature.output,
            network_signature.output,
        )
        self._replay_buffer = replay_buffers.HierarchicalReplayBuffer(
            datapoint_sig,
            capacity=replay_buffer_capacity,
            hierarchy_depth=len(replay_buffer_sampling_hierarchy),
        )
        self._sampling_hierarchy = replay_buffer_sampling_hierarchy

    def add_episode(self, episode):
        batch = (
            self._input_fn(episode),  # input
            self._target_fn(episode),  # target
            self._mask_fn(episode),  # mask
        )
        buckets = [
            getattr(episode, bucket_name)
            for bucket_name in self._sampling_hierarchy
        ]
        self._replay_buffer.add(batch, buckets)

    def train_epoch(self, network):
        def data_stream():
            for _ in range(self._n_steps_per_epoch):
                yield self._replay_buffer.sample(self._batch_size)

        start_time = time.time()
        metrics = network.train(data_stream, self._n_steps_per_epoch)
        metrics['time'] = time.time() - start_time
        return metrics

    def save(self, path):
        # The only training state that we care about is the replay buffer.
        # It gets pretty large with visual observations, so we compress it.
        compression.dump(self._replay_buffer, path)

    def restore(self, path):
        self._replay_buffer = compression.load(path)
