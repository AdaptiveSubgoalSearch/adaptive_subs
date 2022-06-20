"""Temporal difference trainer."""

import collections

import gin
import numpy as np

from alpacka import data
from alpacka.trainers import base
from alpacka.trainers import replay_buffers
from alpacka.utils import compression


TDTargetData = collections.namedtuple(
    'TDTargetData', ['cum_reward', 'bootstrap_obs', 'bootstrap_gamma'])


@gin.configurable
def target_n_return(episode, n, gamma):
    """Calculates targets with n-step bootstraped returns.

    It returns three vectors cum_rewards, bootstrap_obs, bootstrap_gammas
    of the same length as the given episode (ep_len).

    Typically (a):
        cum_rewards[i] = r_i + ... + gamma^{n-1} r_{i+n-1},
        bootstrap_obs[i] is the observation at step i+n,
        bootstrap_gammas[i] = gamma^n.

    Things are adjusted towards the end of the episode (b):
        cum_rewards[i] = r_i + ... + r_{ep_len-1},
        bootstrap_obs[i] is the observation at step ep_len,
        bootstrap_gammas[i] = gamma^{ep_len - i} if not done else 0.

    These values are used to calculate, td targets during training:
        trarget[i] = cum_rewards[i] +  bootstrap_gammas[i]
                                       * value(bootstrap_obs[i]).

    Which for case (a) is the usual TD(n) target:
        trarget[i] = r_i + ... + gamma^{n-1} r_{i+n-1} +  gamma^n
                                                         * value(obs[i+n]).
    """
    assert n >= 1, 'TD(0) does not make sense'
    truncated = episode.truncated
    rewards = episode.transition_batch.reward
    ep_len = len(episode.transition_batch.observation)
    cum_reward = 0

    cum_rewards = []
    bootstrap_obs = []
    bootstrap_gammas = []
    for curr_obs_index in reversed(range(ep_len)):
        reward_to_remove = (0 if curr_obs_index + n >= ep_len
            else gamma ** (n - 1) * rewards[curr_obs_index + n])
        # cum_rewards calculated progressively
        # (for efficiency reasons when n is big)
        cum_reward -= reward_to_remove
        cum_reward *= gamma
        cum_reward += rewards[curr_obs_index]

        bootstrap_index = min(curr_obs_index + n, ep_len)
        bootstrap_ob = \
            episode.transition_batch.next_observation[bootstrap_index - 1]
        bootstrap_gamma = gamma ** (bootstrap_index - curr_obs_index)
        if bootstrap_index == ep_len and not truncated:
            bootstrap_gamma = 0

        cum_rewards.append(cum_reward)
        bootstrap_obs.append(bootstrap_ob)
        bootstrap_gammas.append(bootstrap_gamma)

    datapoints = TDTargetData(cum_reward=np.array(cum_rewards,
                                            dtype=np.float)[::-1, np.newaxis],
                              bootstrap_obs=np.array(bootstrap_obs)[::-1],
                              bootstrap_gamma=np.array(bootstrap_gammas,
                                            dtype=np.float)[::-1, np.newaxis])
    return datapoints


class TDTrainer(base.Trainer):
    """TDTrainer trainer.

    Trains the network based using temporal difference targets
    sampled from a replay buffer.
    """

    def __init__(
        self,
        network_signature,
        temporal_diff_n,
        gamma=1.0,
        batch_size=64,
        n_steps_per_epoch=1000,
        replay_buffer_capacity=1000000,
        replay_buffer_sampling_hierarchy=(),
        polyak_coeff=None
    ):
        """Initializes TDTrainer.

        Args:
            network_signature (pytree): Input signature for the network.
            temporal_diff_n: temporal difference distance, np.inf is supported
            gamma: discount rate
            batch_size (int): Batch size.
            n_steps_per_epoch (int): Number of optimizer steps to do per
                epoch.
            replay_buffer_capacity (int): Maximum size of the replay buffer.
            replay_buffer_sampling_hierarchy (tuple): Sequence of Episode
                attribute names, defining the sampling hierarchy.
            polyak_coeff: polyak averaging coefficient
        """
        super().__init__(network_signature)
        target = lambda episode: target_n_return(episode, temporal_diff_n,
                                                 gamma)
        self._target_fn = lambda episode: data.nested_map(
            lambda f: f(episode), target
        )
        self._batch_size = batch_size
        self._n_steps_per_epoch = n_steps_per_epoch

        td_target_signature = TDTargetData(
            cum_reward=network_signature.output,
            bootstrap_obs=network_signature.input,
            bootstrap_gamma=network_signature.output)
        datapoint_sig = (network_signature.input, td_target_signature)
        self._replay_buffer = replay_buffers.HierarchicalReplayBuffer(
            datapoint_sig,
            capacity=replay_buffer_capacity,
            hierarchy_depth=len(replay_buffer_sampling_hierarchy),
        )
        self._sampling_hierarchy = replay_buffer_sampling_hierarchy
        self._polyak_coeff = polyak_coeff
        self._target_network_params = None
        self._target_network = None

    def add_episode(self, episode):
        buckets = [
            getattr(episode, bucket_name)
            for bucket_name in self._sampling_hierarchy
        ]
        self._replay_buffer.add(
            (
                episode.transition_batch.observation,
                self._target_fn(episode),
            ),
            buckets,
        )

    def train_epoch(self, network):
        if self._polyak_coeff is not None:
            current_params = network.params
            if self._target_network_params is None:
                self._target_network_params = current_params
                self._target_network = network.clone()

            target_network_params_ = []
            for target_nn_layer, current_nn_layer in \
                    zip(self._target_network_params, current_params):
                target_network_params_.append(
                    self._polyak_coeff * target_nn_layer +
                    (1.0 - self._polyak_coeff) * current_nn_layer)

            self._target_network_params = target_network_params_
            self._target_network.params = self._target_network_params
        else:
            self._target_network = network

        def data_stream():
            for _ in range(self._n_steps_per_epoch):
                obs, (cum_reward, bootstrap_obs, bootstrap_gamma) = \
                    self._replay_buffer.sample(self._batch_size)
                preds = self._target_network.predict(bootstrap_obs)
                target = cum_reward + bootstrap_gamma * preds
                yield obs, target, np.ones_like(target)

        return network.train(data_stream, self._n_steps_per_epoch)

    def save(self, path):
        # See SupervisedTrainer.save().
        compression.dump(self._replay_buffer, path)

    def restore(self, path):
        self._replay_buffer = compression.load(path)
