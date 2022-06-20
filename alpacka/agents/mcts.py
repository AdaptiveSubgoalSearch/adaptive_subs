"""Base class for Monte Carlo Tree Search algorithms."""

import math

import gin
import numpy as np

from alpacka.agents import tree_search
from alpacka.utils import space as space_utils


@gin.configurable
def puct_exploration_bonus(child_count, parent_count, prior_probability):
    """PUCT exploration bonus.

    A variant with weight changing over time is used in AlphaZero.

    Args:
        child_count (int): Number of visits in the child node so far.
        parent_count (int): Number of visits in the parent node so far.
        prior_probability (float): Prior probability of the child.

    Returns:
        float: Exploration bonus to apply to the child.
    """
    return math.sqrt(parent_count) / (child_count + 1) * prior_probability


class Node(tree_search.Node):
    """Base class for nodes of the MCTS tree.

    Attrs:
        prior_probability (float): Prior probability of choosing this node
            from the parent.
    """

    def __init__(self, prior_probability, state):
        super().__init__(state=state)
        self.prior_probability = prior_probability


class MCTSAgent(tree_search.TreeSearchAgent):
    """Monte Carlo Tree Search base class."""

    def __init__(
        self,
        exploration_bonus_fn=puct_exploration_bonus,
        exploration_weight=1.0,
        sampling_temperature=0.0,
        prior_noise_weight=0.0,
        prior_noise_parameter=1.0,
        **kwargs
    ):
        """Initializes MCTSAgent.

        Args:
            exploration_bonus_fn (callable): Function calculating an
                exploration bonus for a given node. It's added to the node's
                quality when choosing a node to explore in an MCTS pass.
                Signature: (child_count, parent_count, prior_prob) -> bonus.
            exploration_weight (float): Weight of the exploration bonus.
            sampling_temperature (float): Sampling temperature for choosing the
                actions on the real environment.
            prior_noise_weight (float): Weight of the Dirichlet noise added to
                the prior.
            prior_noise_parameter (float): Parameter of the Dirichlet noise
                added to the prior.
            **kwargs: TreeSearch init keyword arguments.
        """
        super().__init__(**kwargs)
        self._exploration_bonus = exploration_bonus_fn
        self._exploration_weight = exploration_weight
        self._sampling_temperature = sampling_temperature
        self._prior_noise_weight = prior_noise_weight
        self._prior_noise_parameter = prior_noise_parameter

    def _choose_action(self, node, actions, exploratory):
        """Chooses the action to take in a given node based on child qualities.

        Args:
            node (Node): Node to choose an action from.
            actions (list): List of allowed actions.
            exploratory (bool): Whether the choice should be exploratory (in
                an MCTS pass) or not (when choosing the final action on the real
                environment).

        Returns:
            Action to take.
        """
        def rate_child(child):
            if exploratory:
                quality = child.quality(self._discount) + (
                    self._exploration_weight * self._exploration_bonus(
                        child.count, node.count, child.prior_probability
                    )
                )
            else:
                quality = np.log(child.count) if child.count else float('-inf')
                # Sample an action to perform on the real environment using
                # Gumbel sampling. No need to normalize logits.
                u = np.random.uniform(low=1e-6, high=1.0 - 1e-6)
                g = -np.log(-np.log(u))
                quality += g * self._sampling_temperature
            return quality

        child_qualities_and_actions = [
            (rate_child(node.children[action]), action) for action in actions
        ]
        (_, action) = max(child_qualities_and_actions)
        return action

    def _on_new_root(self, root):
        """Adds prior noise to the root."""
        prior = np.array([child.prior_probability for child in root.children])
        noise = np.random.dirichlet(
            [self._prior_noise_parameter] * len(root.children)
        )
        prior = (
            (1 - self._prior_noise_weight) * prior +
            self._prior_noise_weight * noise
        )
        for (child, p) in zip(root.children, prior):
            child.prior_probability = p

    def _compute_node_info(self, node):
        node_info = super()._compute_node_info(node)
        qualities = node_info['qualities']
        prior_probabilities = np.array([
            child.prior_probability for child in node.children
        ])
        exploration_bonuses = self._exploration_weight * np.array([
            self._exploration_bonus(
                child.count, node.count, child.prior_probability
            )
            for child in node.children
        ])
        total_scores = qualities + exploration_bonuses
        return {
            **node_info,
            'prior_probabilities': prior_probabilities,
            'exploration_bonuses': exploration_bonuses,
            'total_scores': total_scores,
        }


def uniform_prior(n):
    """Uniform prior distribution on n actions."""
    return np.full(shape=(n,), fill_value=(1 / n))
