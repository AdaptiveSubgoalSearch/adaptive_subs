"""Maximum Entropy Tree Search."""

import enum

import gin
import numpy as np
from scipy import optimize
from scipy import stats

from alpacka import data
from alpacka import math
from alpacka.agents import stochastic_mcts
from alpacka.agents import tree_search
from alpacka.utils import space as space_utils


@gin.configurable
class SoftQualityNetworkNewLeafRater(stochastic_mcts.NewLeafRater):
    """Rates new leaves using a soft Q-network."""

    def __init__(
        self, agent, boltzmann_temperature=None, inject_log_temperature=False
    ):
        super().__init__(
            agent,
            use_policy=False,
            boltzmann_temperature=boltzmann_temperature,
        )
        self._inject_log_temperature = inject_log_temperature

    def qualities(self, observation, model):
        del model

        observations = np.expand_dims(observation, axis=0)
        if self._inject_log_temperature:
            log_temperatures = np.array([[np.log(self._agent.temperature)]])
            inp = (observations, log_temperatures)
        else:
            inp = observations

        result = yield inp

        qualities = result[0]
        return qualities

    def network_signature(self, observation_space, action_space):
        obs_sig = space_utils.signature(observation_space)
        if self._inject_log_temperature:
            input_sig = (obs_sig, data.TensorSignature(shape=(1,)))
        else:
            input_sig = obs_sig

        n_actions = space_utils.max_size(action_space)
        action_vector_sig = data.TensorSignature(shape=(n_actions,))
        output_sig = action_vector_sig
        # Input: observation, output: quality vector and optionally policy
        # vector.
        return data.NetworkSignature(input=input_sig, output=output_sig)


class TemperatureTuner:
    """Base class for temperature tuners."""

    def __init__(self, reference_temperature):
        """Initializes TemperatureTuner.

        Args:
            reference_temperature (float): Reference temperature. The
                temperature should be adjusted around this value.
        """
        self._reference_temperature = reference_temperature

    def __call__(self, qualities):
        """Calculates the temperature based on an array of node qualities.

        Args:
            qualities (np.ndarray): Array of shape (n_nodes, n_actions)
                with node qualities.

        Returns:
            float: The tuned temperature.
        """
        raise NotImplementedError


@gin.configurable
class ConstantTuner(TemperatureTuner):
    """No-op temperature tuner - returns a constant."""

    def __call__(self, qualities):
        del qualities
        return self._reference_temperature


@gin.configurable
class MeanEntropyTuner(TemperatureTuner):
    """Tuner adjusting the temperature to get target mean entropy."""

    def __init__(
        self,
        reference_temperature,
        target_entropy=1.0,
        temperature_range=1000.0,
    ):
        r"""Initializes MeanEntropyTuner.

        Args:
            reference_temperature (float): The middle (in log-space) of the
                allowed temperature range.
            target_entropy (float): The desired mean entropy across nodes.
            temperature_range (float): The half-length (in log-space) of the
                allowed temperature range.
                temperature \in [reference / range, reference * range]
        """
        super().__init__(reference_temperature)
        self._target_entropy = target_entropy
        self._min_temperature = reference_temperature / temperature_range
        self._max_temperature = reference_temperature * temperature_range

    def __call__(self, qualities):
        def entropy_given_temperature(temperature):
            entropies = math.categorical_entropy(
                logits=(qualities / (temperature + 1e-6)), mean=False
            )
            return np.mean(entropies)

        min_entropy = entropy_given_temperature(self._min_temperature)
        max_entropy = entropy_given_temperature(self._max_temperature)

        def less_or_close(a, b):
            return a < b or np.isclose(a, b)

        if less_or_close(self._target_entropy, min_entropy):
            temperature = self._min_temperature
        elif less_or_close(max_entropy, self._target_entropy):
            temperature = self._max_temperature
        else:
            def excess_entropy(log_temperature):
                return entropy_given_temperature(
                    np.exp(log_temperature)
                ) - self._target_entropy

            log_temperature = optimize.brentq(
                excess_entropy,
                a=np.log(self._min_temperature),
                b=np.log(self._max_temperature),
                rtol=0.01,
            )
            temperature = np.exp(log_temperature)

        return temperature


@gin.configurable
class EntropyRangeTuner(TemperatureTuner):
    """Tuner adjusting the temperature to get target mean entropy."""

    def __init__(
        self,
        reference_temperature,
        min_entropy=0.1,
        max_entropy=1.0,
        temperature_range=1000.0,
        temperature_penalty=0.001,
    ):
        r"""Initializes EntropyRangeTuner.

        Args:
            reference_temperature (float): The middle (in log-space) of the
                allowed temperature range.
            min_entropy (float): Minimum allowed entropy.
            max_entropy (float): Maximum allowed entropy.
            temperature_range (float): The half-length (in log-space) of the
                allowed temperature range.
                temperature \in [reference / range, reference * range]
            temperature_penalty (float): Penalty for high temperatures.
        """
        super().__init__(reference_temperature)
        self._min_entropy = min_entropy
        self._max_entropy = max_entropy
        self._min_temperature = reference_temperature / temperature_range
        self._max_temperature = reference_temperature * temperature_range
        self._temperature_penalty = temperature_penalty

    def __call__(self, qualities):
        def discrepancy(log_temperature):
            temperature = np.exp(log_temperature)
            entropies = math.categorical_entropy(
                logits=(qualities / (temperature + 1e-6)), mean=False
            )
            return np.mean(np.maximum(
                np.maximum(
                    self._min_entropy - entropies,
                    entropies - self._max_entropy,
                ),
                0,
            ) + self._temperature_penalty * log_temperature)

        result = optimize.minimize_scalar(
            discrepancy,
            method='bounded',
            bounds=(
                np.log(self._min_temperature),
                np.log(self._max_temperature),
            ),
        )
        return np.exp(result.x)


@gin.configurable
class StandardDeviationTuner(TemperatureTuner):
    """Tuner adjusting the temperature based on the std of qualities."""

    def __init__(self, reference_temperature, temperature_range=1000.0):
        r"""Initializes StandardDeviationTuner.

        Args:
            reference_temperature (float): The middle (in log-space) of the
                allowed temperature range.
            temperature_range (float): The half-length (in log-space) of the
                allowed temperature range.
                temperature \in [reference / range, reference * range]
        """
        super().__init__(reference_temperature)
        self._min_temperature = reference_temperature / temperature_range
        self._max_temperature = reference_temperature * temperature_range

    def __call__(self, qualities):
        std = np.mean(np.std(qualities, axis=-1))
        # We want the reference temperature to work on normalized Qs, so
        # that Q / t = (Q / std(Q)) / ref_t. Hence, t = std(q) * ref_t.
        temperature = std * self._reference_temperature
        # Clip the temperature to avoid numerical issues.
        return np.clip(
            temperature, self._min_temperature, self._max_temperature
        )


@gin.configurable
class SoftIteration:
    """Soft iteration mode, defined by a function Q(s, .) -> V(s).

    For extensibility, we provide access to all information about the given
    node.
    """

    def __call__(self, node, discount):
        """Calculates the value of a node.

        Args:
            node (Node): The node to calculate value for.
            discount (float): Discount factor.

        Returns:
            float: The value of the given node.
        """
        raise NotImplementedError


@gin.configurable
class SoftPolicyIteration:
    """Soft policy iteration, used e.g. in Soft Actor-Critic."""

    def __init__(self, pseudoreward_shaping=1.0):
        """Initializes SoftPolicyIteration.

        Args:
            pseudoreward_shaping (float): Pseudoreward shaping constant. At 0,
                the pseudorewards gained for the policy entropy are always
                positive. At 1, they are always negative. The purpose of
                pseudoreward shaping is to ensure exploration when no reward
                signal is available and the Q-network is not trained yet - let's
                say it gives zero output. In such a case, a positive
                pseudoreward obtained on the first visited path from the root
                will cause the planner to choose this path over and over again,
                which yields no exploration. As such, it's desirable for the
                pseudorewards to be negative.
        """
        self._pseudoreward_shaping = pseudoreward_shaping

    def __call__(self, node, discount):
        count_sum = sum(child.count for child in node.children)

        # Pseudoreward shaping shifts the pseudorewards down by a fraction of
        # the maximum entropy: log(n_actions).
        shift = self._pseudoreward_shaping * np.log(len(node.children))
        pseudorewards = [
            -node.temperature * (np.log(child.count / count_sum) + shift)
            for child in node.children
        ]
        return sum(
            (child.quality(discount) + pseudoreward) * child.count
            for (child, pseudoreward) in zip(node.children, pseudorewards)
        ) / count_sum


@gin.configurable
class SoftQIteration:
    """Soft Q-iteration, used e.g. in Soft Q-Learning."""

    def __call__(self, node, discount):
        return node.temperature * math.log_mean_exp([
            child.quality(discount) / node.temperature
            for child in node.children
        ])


class Node(tree_search.Node):
    """Node of MaxEntTreeSearch."""

    def __init__(self, init_quality, temperature, soft_iteration):
        """Initializes Node.

        Args:
            init_quality (float or None): Quality received from
                the NewLeafRater for this node, or None if it's the root.
            temperature (float): Temperature at the moment of node creation.
            soft_iteration (SoftIteration): Soft iteration mode.
        """
        super().__init__()

        self._init_quality = init_quality
        self._quality = init_quality
        self._reward_sum = 0
        self._reward_count = 0

        self.temperature = temperature
        self._soft_iteration = soft_iteration

    def visit(self, reward, value, discount):
        if reward is None:
            return

        self._reward_sum += reward
        self._reward_count += 1

        self.update(discount, value)

    def update(self, discount, value=None):
        """Recalculates the quality of the node.

        Args:
            discount (float): Discount factor.
            value (float or None): Backpropagated value if the node is updated
                for the first time, None otherwise.
        """
        if not self.is_leaf:
            # In an inner node, recompute the value based on the children's
            # qualities.
            value = self.value(discount)
        elif value is None:
            return

        quality_sum = self._reward_sum + discount * value * self._reward_count
        quality_count = self._reward_count

        if self._init_quality is not None:
            # No reward is given to a new leaf, so the initial quality needs to
            # be added separately.
            quality_sum += self._init_quality
            quality_count += 1

        self._quality = quality_sum / quality_count

    def quality(self, discount):
        del discount
        return self._quality

    @property
    def count(self):
        return self._reward_count + int(self._init_quality is not None)

    def value(self, discount):
        """Calculates the value according to the chosen soft iteration mode."""
        return self._soft_iteration(self, discount)


@gin.constants_from_enum
class QualityAccumulation(enum.Enum):
    """Quality accumulation mode."""

    # Accumulation only in the subtree of the current top-level node.
    subtree = 0
    # Accumulation across all nodes.
    full_tree = 1


@gin.constants_from_enum
class QualityRecalculation(enum.Enum):
    """Quality recalculation mode."""

    # No recalculation.
    none = 0
    # Recalculation only in the subtree of the current top-level node.
    subtree = 1
    # Recalculation in all nodes.
    full_tree = 2


@gin.constants_from_enum
class InitQuality(enum.Enum):
    """Node quality initialization based on the output of NewLeafRater."""

    # Take the qualities.
    quality = 0
    # Take a logarithm of the prior.
    log_prior = 1


class TargetPolicy:
    """The policy we want to match in the nodes."""

    def __call__(self, optimal_policy, node):
        """Computes the target policy.

        Args:
            optimal_policy (np.ndarray): The optimal policy distribution.
            node (Node): The current node.

        Returns:
            np.ndarray: The target policy distribution.
        """
        raise NotImplementedError


@gin.configurable
class OptimalTargetPolicy(TargetPolicy):
    """The simplest target policy: same as the optimal policy."""

    def __call__(self, optimal_policy, node):
        del node
        return optimal_policy


@gin.configurable
class MentsTargetPolicy(TargetPolicy):
    """Target policy used by MENTS.

    Combined with sampling actions with temperature 1, gives the E2W sampling
    strategy implemented in MENTS.
    """

    def __init__(self, epsilon=1.0):
        self._epsilon = epsilon

    def __call__(self, optimal_policy, node):
        n_actions = len(optimal_policy)
        exploration_decay = np.clip(
            self._epsilon * n_actions / np.log(node.count + 2), 0, 1
        )
        return (
            (1 - exploration_decay) * optimal_policy +
            exploration_decay * np.ones(n_actions) / n_actions
        )


class MaxEntTreeSearchAgent(tree_search.TreeSearchAgent):
    """Maxmum Entropy Tree Search."""

    def __init__(
        self,
        new_leaf_rater_class=stochastic_mcts.RolloutNewLeafRater,
        temperature_tuner_class=ConstantTuner,
        soft_iteration_class=SoftPolicyIteration,
        reference_temperature=1.0,
        model_selection_temperature=1e-3,
        real_selection_temperature=1e-3,
        model_selection_tolerance=0.0,
        quality_accumulation=QualityAccumulation.subtree,
        quality_recalculation=QualityRecalculation.none,
        log_temperature_decay=0.9,
        init_quality=InitQuality.quality,
        target_policy_class=OptimalTargetPolicy,
        **kwargs
    ):
        """Initializes MaxEntTreeSearchAgent.

        Args:
            new_leaf_rater_class (type): NewLeafRater for estimating qualities
                of new leaves.
            temperature_tuner_class (type): TemperatureTuner for adjusting the
                temperature based on node qualities.
            soft_iteration_class (type): Soft iteration mode.
            reference_temperature (float): Reference value for temperature
                tuning. The semantics is dependent on temperature_tuner_class.
            model_selection_temperature (float): Temperature of action selection
                in the model. The default is 0.001: choose greedily with random
                tie breaking. Can be set to higher values to enable in-tree
                parallelism.
            real_selection_temperature (float): Temperature of action selection
                in the real environment. The default is 0.001: choose greedily
                with random tie breaking. Can be set to higher values for more
                diversity in the collected episodes, to enable closed-loop
                training.
            model_selection_tolerance (float): Determines how often an
                overexplored action can be chosen in the model. 0 - never,
                1 - always. Can take values in between.
            quality_accumulation (QualityAccumulation): The set of nodes over
                which to accumulate qualities for temperature tuning. By
                default, accumulate over the subtree of the current top-level
                node.
            quality_recalculation (QualityRecalculation): The set of nodes for
                which to recalculate qualities after temperature tuning. By
                default, don't recalculate.
            log_temperature_decay (float): Decay term for the exponential moving
                average over the tuned log-temperature.
            init_quality (InitQuality): Node quality initialization mode.
            target_policy_class (type): TargetPolicy we want to match in the
                nodes.
            kwargs: TreeSearchAgent.__init__ keyword arguments.
        """
        super().__init__(**kwargs)
        self._new_leaf_rater = new_leaf_rater_class(self)
        self._temperature_tuner = temperature_tuner_class(reference_temperature)
        self._soft_iteration = soft_iteration_class()
        self._reference_temperature = reference_temperature
        self._model_selection_temperature = model_selection_temperature
        self._real_selection_temperature = real_selection_temperature
        self._model_selection_tolerance = model_selection_tolerance
        self._quality_accumulation = quality_accumulation
        self._quality_recalculation = quality_recalculation
        self._log_temperature_decay = log_temperature_decay
        self._init_quality = init_quality
        self._target_policy = target_policy_class()

        self._tuned_log_temperature = np.log(reference_temperature)
        self._initial_root = None

    def _choose_action(self, node, actions, exploratory):
        """Chooses the action to take in a given node based on child qualities.

        Args:
            node (Node): Node to choose an action from.
            actions (list): List of allowed actions.
            exploratory (bool): Whether the choice should be exploratory (in
                a planning pass) or not (when choosing the final action on the
                real environment).

        Returns:
            Action to take.
        """
        if exploratory:
            selection_tolerance = self._model_selection_tolerance
            selection_temperature = self._model_selection_temperature
        else:
            # Always sample out of all actions in the real env. The visit counts
            # are not included in the training data, so it doesn't make sense to
            # filter out overexplored actions.
            selection_tolerance = 1.0
            selection_temperature = self._real_selection_temperature

        qualities = np.array([
            node.children[action].quality(self._discount)
            for action in actions
        ])

        if not exploratory:
            # In top-level nodes, re-tune the temperature based on qualities
            # of a defined set of nodes.
            acc_qualities = self._accumulate_qualities(node)
            if acc_qualities.size > 0:
                temperature = self._temperature_tuner(acc_qualities)
                self._update_temperature(temperature)
                self._recalculate_qualities(node)

        qualities = np.array([
            node.children[action].quality(self._discount)
            for action in actions
        ])

        # Calculate the mismatch between the empirical policy resulting from
        # current child visit counts, and the target policy resulting from the
        # softmax of qualities scaled by the temperature.
        node.temperature = self.temperature
        optimal_policy = math.softmax(qualities / node.temperature)
        target_policy = self._target_policy(optimal_policy, node)
        counts = np.array([node.children[action].count for action in actions])
        # Use a lower epsilon than in action filtering to avoid a corner case
        # with no underexplored actions.
        empirical_policy = counts / (np.sum(counts) + 1e-9)
        policy_mismatch = target_policy - empirical_policy

        # Shift the mismatch up according to selection_tolerance to allow
        # choosing some of the overexplored actions.
        tolerated_mismatch = (
            policy_mismatch + selection_tolerance * empirical_policy
        )
        # Filter actions with positive mismatch.
        logits_and_actions = [
            (np.log(mismatch + 1e-6), action)
            for (mismatch, action) in zip(tolerated_mismatch, actions)
            if mismatch >= -1e-6
        ]
        # Sample an action according to selection_temperature.
        (logits, actions) = zip(*logits_and_actions)
        action = math.categorical_sample(
            logits=(np.array(logits) / selection_temperature)
        )
        return actions[action]

    def _accumulate_qualities(self, node):
        acc_mode = self._quality_accumulation
        if acc_mode is QualityAccumulation.subtree:
            acc_root = node
        elif acc_mode is QualityAccumulation.full_tree:
            acc_root = self._initial_root
        else:
            raise TypeError(f'Invalid quality accumulation mode: {acc_mode}.')

        def accumulate(node, acc):
            qualities = [
                child.quality(self._discount)
                for child in node.children
            ]
            if qualities:
                acc.append(qualities)
            for child in node.children:
                accumulate(child, acc)

        acc_qualities = []
        accumulate(acc_root, acc_qualities)
        return np.array(acc_qualities)

    def _update_temperature(self, temperature):
        decay = self._log_temperature_decay
        self._tuned_log_temperature = (
            decay * self._tuned_log_temperature +
            (1 - decay) * np.log(temperature)
        )

    @property
    def temperature(self):
        return np.exp(self._tuned_log_temperature)

    def _recalculate_qualities(self, node):
        recalc_mode = self._quality_recalculation
        if recalc_mode is QualityRecalculation.none:
            return

        recalc_mode = self._quality_recalculation
        if recalc_mode is QualityRecalculation.subtree:
            recalc_root = node
        elif recalc_mode is QualityRecalculation.full_tree:
            recalc_root = self._initial_root
        else:
            raise TypeError(
                f'Invalid quality recalculation mode: {recalc_mode}.'
            )

        def update(node):
            for child in node.children:
                update(child)
            node.temperature = np.exp(self._tuned_log_temperature)
            node.update(self._discount)

        update(recalc_root)

    def reset(self, env, observation):
        yield from super().reset(env, observation)
        self._initial_root = self._root
        self._tuned_log_temperature = np.log(self._reference_temperature)

    def _init_root_node(self, state):
        return self._init_node(init_quality=None)

    def _init_child_nodes(self, leaf, observation):
        del leaf
        child_qualities_and_probs = yield from self._new_leaf_rater(
            observation, self._model
        )

        (qualities, prior) = zip(*child_qualities_and_probs)
        if self._init_quality is InitQuality.quality:
            pass
        elif self._init_quality is InitQuality.log_prior:
            qualities = np.log(prior)
        else:
            raise TypeError(
                f'Invalid quality initialization: {self._init_quality}.'
            )

        return list(map(self._init_node, qualities))

    def _init_node(self, init_quality):
        return Node(
            init_quality=init_quality,
            temperature=np.exp(self._tuned_log_temperature),
            soft_iteration=self._soft_iteration,
        )

    def network_signature(self, observation_space, action_space):
        # Delegate defining the network signature to NewLeafRater. This is the
        # only part of the agent that uses a network, so it should decide what
        # sort of network it needs.
        return self._new_leaf_rater.network_signature(
            observation_space, action_space
        )

    def _compute_node_info(self, node):
        info = super()._compute_node_info(node)
        softmax_policy = math.softmax(info['qualities'] / node.temperature)
        policy_mismatch = softmax_policy - info['action_histogram']
        return {
            'temperature': node.temperature,
            'softmax_policy': softmax_policy,
            'policy_mismatch': policy_mismatch,
            **info
        }

    @classmethod
    def compute_metrics(cls, episodes):
        metrics = super().compute_metrics(episodes)

        temperatures = np.array([
            temp
            for episode in episodes
            for temp in episode.transition_batch.agent_info['temperature']
        ], dtype=np.float)

        return {
            'temperature_gmean': stats.gmean(temperatures),
            'temperature_min': np.min(temperatures),
            'temperature_max': np.max(temperatures),
            **metrics
        }
