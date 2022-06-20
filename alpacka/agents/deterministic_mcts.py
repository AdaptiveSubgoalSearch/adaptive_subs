"""Monte Carlo Tree Search for deterministic environments."""

import gin
import numpy as np

from alpacka import data
from alpacka.agents import mcts
from alpacka.utils import space as space_utils


class ValueTraits:
    """Value traits base class.

    Defines constants for abstract value types.
    """

    zero = None
    dead_end = None


@gin.configurable
class ScalarValueTraits(ValueTraits):
    """Scalar value traits.

    Defines constants for the most basic case of scalar values.
    """

    zero = 0.0

    def __init__(self, dead_end_value=-2.0):
        self.dead_end = dead_end_value


class ValueAccumulator:
    """Value accumulator base class.

    Accumulates abstract values for a given node across multiple MCTS passes.
    """

    def __init__(self, value):
        # Creates and initializes with typical add
        self.add(value)

    def add(self, value):
        """Adds an abstract value to the accumulator.

        Args:
            value: Abstract value to add.
        """
        raise NotImplementedError

    def get(self):
        """Returns the accumulated abstract value for backpropagation."""
        raise NotImplementedError

    def index(self):
        """Returns an index for selecting the best node."""
        raise NotImplementedError

    def target(self):
        """Returns a target for value function training."""
        raise NotImplementedError

    def count(self):
        """Returns the number of accumulated values."""
        raise NotImplementedError


@gin.configurable
class ScalarValueAccumulator(ValueAccumulator):
    """Scalar value accumulator.

    Calculates a mean over accumulated values and returns it as the
    backpropagated value, node index and target for value network training.
    """

    def __init__(self, value):
        self._sum = 0.0
        self._count = 0
        super().__init__(value)

    def add(self, value):
        self._sum += value
        self._count += 1

    def get(self):
        return self._sum / self._count

    def index(self):
        return self.get()

    def target(self):
        return self.get()

    def count(self):
        return self._count


class Node(mcts.Node):
    """Node of DeterministicMCTS."""

    def __init__(self, reward, value_acc, prior_probability,
                 deterministic_rewards):
        """Initializes Node.

        Args:
            reward (float or None): Reward obtained when stepping into the node,
                or None if it's the root.
            value_acc (ValueAccumulator): ValueAccumulator of the node.
            prior_probability (float): Prior probability of picking this node
                from its parent.
            deterministic_rewards (bool): Check that reward doesn't change
                over all visits.
        """
        super().__init__(prior_probability)
        self._reward = reward
        self._count = 0
        self._value_acc = value_acc
        self._deterministic_rewards = deterministic_rewards

    @property
    def value_acc(self):
        return self._value_acc

    def visit(self, reward, value, discount):
        del discount

        if self._reward is None:
            self._reward = reward
        elif reward is not None and self._deterministic_rewards:
            assert reward == self._reward, (
                'Nondeterministic rewards: {} and {}.'.format(
                    self._reward, reward
                )
            )
        self._value_acc.add(value)
        self._count += 1

    def quality(self, discount):
        return self._reward + discount * self._value_acc.get()

    @property
    def count(self):
        return self._count


class DeterministicMCTSAgent(mcts.MCTSAgent):
    """Monte Carlo Tree Search for deterministic environments."""

    def __init__(
        self,
        graph_mode=True,
        avoid_loops=True,
        value_traits_class=ScalarValueTraits,
        value_accumulator_class=ScalarValueAccumulator,
        **kwargs
    ):
        """Initializes DeterministicMCTSAgent.

        Args:
        """
        super().__init__(**kwargs)
        self._graph_mode = graph_mode
        self._avoid_loops = avoid_loops
        self._value_traits = value_traits_class()
        self._value_acc_class = value_accumulator_class
        self._state_to_value_acc = {}
        self._model_visited = set()
        self._real_visited = set()

    def _init_root_node(self, state):
        return Node(
            reward=None,
            value_acc=self._init_value_acc(state, value=0),
            prior_probability=(
                1 / space_utils.max_size(self._model.action_space)
            ),
            deterministic_rewards=self._model_class.is_perfect,
        )

    def _before_pass(self):
        self._model_visited = set()

    def _before_model_step(self, node):
        if self._avoid_loops:
            self._model_visited.add(node.value_acc)

    def _before_real_step(self, node):
        self._real_visited = {node.value_acc} if self._avoid_loops else set()

    def _make_filter_fn(self, exploratory):
        visited = self._model_visited if exploratory else self._real_visited
        return lambda node: node.value_acc not in visited

    @property
    def _zero_quality(self):
        return self._value_traits.zero

    @property
    def _dead_end_quality(self):
        return self._value_traits.dead_end

    def _init_value_acc(self, state, value):
        if self._graph_mode:
            if state not in self._state_to_value_acc:
                value_acc = self._value_acc_class(value)
                self._state_to_value_acc[state] = value_acc
            return self._state_to_value_acc[state]
        else:
            return self._value_acc_class(value)

    def _init_child_nodes(self, leaf, observation):
        actions = list(space_utils.element_iter(self._model.action_space))
        (observations, rewards, dones, states) = \
            yield from self._model.predict_steps(actions, include_state=True)
        # Run the network to predict values for children.
        values = yield observations
        # (batch_size, 1) -> (batch_size,)
        values = np.reshape(values, -1)
        values *= (1 - np.array(dones))

        prior_probs = mcts.uniform_prior(
            space_utils.max_size(self._model.action_space)
        )
        return [
            Node(
                reward,
                self._init_value_acc(state, value),
                prior_probability=prior_prob,
                deterministic_rewards=self._model_class.is_perfect
            )
            for (state, reward, value, prior_prob) in zip(
                states, rewards, values, prior_probs
            )
        ]

    def network_signature(self, observation_space, action_space):
        return data.NetworkSignature(
            input=space_utils.signature(observation_space),
            output=data.TensorSignature(shape=(1,)),
        )
