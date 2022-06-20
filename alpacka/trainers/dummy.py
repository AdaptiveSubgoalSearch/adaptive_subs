"""Dummy Trainer for testing."""

import gin

from alpacka.trainers import base


@gin.configurable
class DummyTrainer(base.Trainer):
    """Dummy Trainer for testing and use with plain Networks (not trainable)."""

    def __init__(self, network_signature):
        super().__init__(network_signature)
        self.replay_buffer = ''  # Dummy replay buffer for testing.

    def add_episode(self, episode):
        del episode

    def train_epoch(self, network):
        return {}

    def save(self, path):
        with open(path, 'w') as f:
            f.write(self.replay_buffer)

    def restore(self, path):
        with open(path, 'r') as f:
            self.replay_buffer = f.read()
