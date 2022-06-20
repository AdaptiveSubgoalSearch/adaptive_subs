"""Base class for trainers."""


class Trainer:
    """Base class for trainers.

    Trainer is something that can train a neural network using data from memory.
    In the most basic setup, it just samples data from a replay buffer. By
    abstracting Trainer out, we can also support other setups, e.g. tabular
    learning on a tree.
    """

    def __init__(self, network_signature):
        """No-op constructor just to specify the interface.

        Args:
            network_signature (pytree): Input signature for the network.
        """
        del network_signature

    def add_episode(self, episode):
        """Adds an episode to memory.

        Args:
            episode (Agent/Trainer-specific): Episode object summarizing the
                collected data for training the TrainableNetwork.
        """
        raise NotImplementedError

    def train_epoch(self, network):
        """Runs one epoch of training.

        Args:
            network (TrainableNetwork): TrainableNetwork instance to be trained.
        """
        raise NotImplementedError

    def save(self, path):
        """Saves the training state.

        Args:
            path (str): Path to save the state at.
        """
        raise NotImplementedError

    def restore(self, path):
        """Restores the training state.

        Args:
            path (str): Path to load the state from.
        """
        raise NotImplementedError
