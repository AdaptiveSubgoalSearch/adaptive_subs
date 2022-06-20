"""Trainer for the multiple network setup."""

import os

import gin

from alpacka import data
from alpacka.networks import core
from alpacka.trainers import base
from alpacka.utils import transformations


class UnionTrainer(base.Trainer):
    """Dedicated trainer for UnionNetwork.

    Stores a separate trainer for every network contained by a corresponding
    UnionNetwork. So every network has its own replay buffer.

    It doesn't work with any other network.
    """
    def __init__(self, network_signature, request_to_trainer=gin.REQUIRED):
        """
        Args:
            network_signature (dict(type -> NetworkSignature)):
                For every request type specifies the signature of a network
                associated with requests of that type. Key: type of request,
                value: NetworkSignature.
            request_to_trainer (dict(type -> callable)):
                For every request type specifies a separate trainer to train
                network associated with requests of that type.
                Key: type of request, value: function to create a trainer.
        """
        super().__init__(network_signature)

        [trainers, signatures] = [
            transformations.map_dict_keys(dictionary, data.request_type_id)
            for dictionary in [request_to_trainer, network_signature]
        ]

        self._trainers = {
            type_id: trainer_fn(sig)
            for type_id, (trainer_fn, sig) in transformations.zip_dicts_strict(
                trainers, signatures
            ).items()
        }

    @property
    def subtrainers(self):
        return self._trainers

    def add_episode(self, episode):
        for trainer in self._trainers.values():
            trainer.add_episode(episode)

    def train_epoch(self, network):
        if not isinstance(network, core.UnionNetwork):
            raise TypeError('UnionTrainer can train only a UnionNetwork.')

        if network.subnetworks.keys() != self._trainers.keys():
            raise TypeError(
                'Mismatch on request types with the given UnionNetwork.\n'
                f'{network.subnetworks.keys()} != {self._trainers.keys()}'
            )

        all_metrics = {}
        for type_id, trainer in self._trainers.items():
            metrics = trainer.train_epoch(network.subnetworks[type_id])
            all_metrics.update({
                f'{type_id}/{key}': value
                for key, value in metrics.items()
            })

        return all_metrics

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        for (slug, trainer) in self.subtrainers.items():
            trainer.save(os.path.join(path, slug))

    def restore(self, path):
        for (slug, trainer) in self.subtrainers.items():
            trainer.restore(os.path.join(path, slug))
