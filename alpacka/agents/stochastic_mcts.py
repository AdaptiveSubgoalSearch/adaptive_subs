"""Monte Carlo Tree Search for stochastic environments."""

import gin
import numpy as np

from alpacka import data
from alpacka import math
from alpacka.agents import core
from alpacka.agents import mcts
from alpacka.agents.tree_search import DeadEnd
from alpacka.utils import space as space_utils


class NewLeafRater:
    """Base class for rating the children of an expanded leaf."""

    def __init__(
        self, agent, use_policy=False, boltzmann_temperature=None
    ):
        """Initializes NewLeafRater.

        Args:
            agent (alpacka.agents.Agent): Agent using the rater.
            use_policy (bool): Whether the prior distribution should be
                calculated as an output of a separate policy network.
            boltzmann_temperature (float or None): If provided, the prior is
                calculated as a Boltzmann distribution parametrized by the
                qualities, and this parameter is interpreted as the temperature
                of that distribution.
                Cannot be provided if use_policy is True.
                If not provided and use_policy is False, the prior is set to
                a uniform distribution.
        """
        self._agent = agent
        assert not (use_policy and boltzmann_temperature)
        assert use_policy
        self._use_policy = use_policy
        self._boltzmann_temperature = boltzmann_temperature

    def __call__(self, observation, model=None):
        """Rates the children of an expanded leaf.

        Args:
            observation (np.ndarray): Observation received at leaf.
            model (EnvModel): Environment model.

        Yields:
            Network prediction requests.

        Returns:
            list: List of pairs (quality, prob) for each action, where quality
                is the estimated quality of the action and prob is its prior
                probability (e.g. from a policy network).
        """
        if self._use_policy:
            (qualities, prior, states, child_infos) = self.qualities_and_prior(
                observation, model
            )
        else:
            assert False
            qualities, states, child_infos = self.qualities(observation, model)
            prior = mcts.uniform_prior(len(qualities))

        if self._boltzmann_temperature is not None:
            prior = math.softmax(
                np.array(qualities) / self._boltzmann_temperature
            )

        return list(zip(qualities, prior, states, child_infos))

    def qualities(self, observation, model):
        """Calculates qualities of children of an expanded leaf.

        Only called when use_policy is False.

        Args:
            observation (np.ndarray): Observation received at leaf.
            model (EnvModel): Environment model.

        Yields:
            Network prediction requests.

        Returns:
            Sequence of child qualities.
        """
        raise NotImplementedError

    def qualities_and_prior(self, observation, model):
        """Calculates qualities and prior over children of an expanded leaf.

        Only called when use_policy is True.

        Args:
            observation (np.ndarray): Observation received at leaf.
            model (EnvModel): Environment model.

        Yields:
            Network prediction requests.

        Returns:
            Pair (qualities, prior), where both elements are sequences of
            floats: child qualities and a prior distribution over children.
        """
        raise NotImplementedError

    def network_signature(self, observation_space, action_space):
        """Defines the signature of networks used by this NewLeafRater.

        Args:
            observation_space (gym.Space): Environment observation space.
            action_space (gym.Space): Environment action space.

        Returns:
            NetworkSignature or None: Either the network signature or None if
            the NewLeafRater doesn't use a network.
        """
        raise NotImplementedError


@gin.configurable
class RolloutNewLeafRater(NewLeafRater):
    """Rates new leaves using rollouts with an Agent."""

    def __init__(
        self,
        agent,
        boltzmann_temperature=None,
        rollout_agent_class=core.RandomAgent,
        rollout_time_limit=100,
    ):
        super().__init__(
            agent,
            use_policy=False,
            boltzmann_temperature=boltzmann_temperature,
        )
        self._discount = agent.discount
        self._rollout_agent = rollout_agent_class()
        self._time_limit = rollout_time_limit

    def qualities(self, observation, model):
        init_state = model.clone_state()

        child_qualities = []
        for init_action in space_utils.element_iter(model.action_space):
            (observation, init_reward, done) = \
                yield from model.step(init_action)
            yield from self._rollout_agent.reset(model, observation)
            value = 0
            total_discount = 1
            time = 0
            while not done and time < self._time_limit:
                (action, _) = yield from self._rollout_agent.act(observation)
                (observation, reward, done) = yield from model.step(action)
                value += total_discount * reward
                total_discount *= self._discount
                time += 1
            child_qualities.append(init_reward + self._discount * value)
            model.restore_state(init_state)
        return child_qualities

    def network_signature(self, observation_space, action_space):
        return self._rollout_agent.network_signature(
            observation_space, action_space
        )


@gin.configurable
class ValueNetworkNewLeafRater(NewLeafRater):
    """Rates new leaves using a value network."""

    def __init__(
        self, agent, value_function, goal_builder,
        use_policy=False, boltzmann_temperature=None
    ):
        assert use_policy, 'Not supported'
        super().__init__(
            agent,
            use_policy=use_policy,
            boltzmann_temperature=boltzmann_temperature,
        )
        self._discount = agent.discount
        self._value_function = value_function
        self._goal_builder = goal_builder

    def qualities(self, observation, model):
        raise ValueError()

    def qualities_and_prior(self, observation, model):
        states_infos_and_prior = self._goal_builder.build_goals(observation)
        if not states_infos_and_prior:
            raise DeadEnd
        (states, child_infos, prior) = zip(*states_infos_and_prior)

        values = self._value_function(states)
        qualities = [
            # Compute the final qualities, masking out the "done" states.
            info.reward + self._discount * value * (1 - info.done)
            for value, info in zip(values, child_infos)
        ]
        return (qualities, prior, states, child_infos)

    def network_signature(self, observation_space, action_space):
        n_actions = space_utils.max_size(action_space)
        if self._use_policy:
            return data.NetworkSignature(
                input=space_utils.signature(observation_space),
                output=(
                    data.TensorSignature(shape=(1,)),
                    data.TensorSignature(shape=(n_actions,))
                ),
            )
        else:
            # Input: observation, output: scalar value.
            return data.NetworkSignature(
                input=space_utils.signature(observation_space),
                output=data.TensorSignature(shape=(1,)),
            )


@gin.configurable
class QualityNetworkNewLeafRater(NewLeafRater):
    """Rates new leaves using a Q-network."""

    def qualities_and_prior(self, observation, model):
        del model
        (qualities, prior) = yield np.expand_dims(observation, axis=0)
        qualities = np.squeeze(qualities, axis=0)
        prior = np.squeeze(prior, axis=0)
        return (qualities, prior)

    def qualities(self, observation, model):
        del model
        qualities = yield np.expand_dims(observation, axis=0)
        return np.squeeze(qualities, axis=0)

    def network_signature(self, observation_space, action_space):
        n_actions = space_utils.max_size(action_space)
        action_vector_sig = data.TensorSignature(shape=(n_actions,))
        if self._use_policy:
            output_sig = (action_vector_sig,) * 2
        else:
            output_sig = action_vector_sig
        # Input: observation, output: quality vector and optionally policy
        # vector.
        return data.NetworkSignature(
            input=space_utils.signature(observation_space),
            output=output_sig,
        )


class Node(mcts.Node):
    """Node of StochasticMCTS."""

    def __init__(self, init_quality, prior_probability, state):
        """Initializes Node.

        Args:
            init_quality (float or None): Quality received from
                the NewLeafRater for this node, or None if it's the root.
            prior_probability (float): Prior probability of picking this node
                from its parent.
        """
        super().__init__(prior_probability=prior_probability, state=state)
        if init_quality is None:
            self._quality_sum = 0
            self._quality_count = 0
        else:
            self._quality_sum = init_quality
            self._quality_count = 1

    def visit(self, reward, value, discount):
        if reward is None:
            return

        quality = reward + discount * value
        self._quality_sum += quality
        self._quality_count += 1

    def quality(self, discount):
        del discount
        return self._quality_sum / self._quality_count

    @property
    def count(self):
        return self._quality_count


class StochasticMCTSAgent(mcts.MCTSAgent):
    """Monte Carlo Tree Search for stochastic environments."""

    def __init__(
        self, value_function, goal_builder, **kwargs,
    ):
        """Initializes StochasticMCTSAgent.

        Args:
            new_leaf_rater_class (type): NewLeafRater for estimating qualities
                of new leaves.
            value_function: Callable(state -> float)
            goal_builder: GoalBuilder
            kwargs: OnlineAgent init keyword arguments.
        """
        super().__init__(**kwargs)
        self._new_leaf_rater = ValueNetworkNewLeafRater(
            agent=self,
            use_policy=True,
            value_function=value_function,
            goal_builder=goal_builder,
        )

    def _init_root_node(self, state):
        return Node(init_quality=None, prior_probability=None, state=state)

    def _init_child_nodes(self, leaf, observation):
        del leaf
        child_qualities_probs_states_and_infos = \
            self._new_leaf_rater(observation)
        return ([
            Node(quality, prob, state=state)
            for (quality, prob, state, _) in child_qualities_probs_states_and_infos
        ], [
            child_info for (_, _, _, child_info) in child_qualities_probs_states_and_infos
        ])

    def network_signature(self, observation_space, action_space):
        # Delegate defining the network signature to NewLeafRater. This is the
        # only part of the agent that uses a network, so it should decide what
        # sort of network it needs.
        return self._new_leaf_rater.network_signature(
            observation_space, action_space
        )
