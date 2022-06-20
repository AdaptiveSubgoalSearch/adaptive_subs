"""Base class for tree search planning algorithms."""

import collections

import numpy as np

from alpacka.agents import base
from alpacka.agents import models
from alpacka.data import ops
from alpacka.utils import metric as metric_utils
from alpacka.utils import space as space_utils


ChildInfo = collections.namedtuple('ChildInfo', [
    'action_list',
    'reward',
    'done',
])


class GoalBuilder:
    def build_goals(self, state):
        """Proposes goals for a given state.

        Goals may be one or more actions ahead.

        Goal builder is also responsible for checking, that proposed goals
        are correct and reachable - MCTS doesn't check that at all.

        Returns:
            List of tuples:
            * child_state (env-dependent): next state of environment
            * child_info (ChildInfo): transition-related information
            * child_probability (float): probability of choosing this
               child in MCTS pass (you can obtain it from network prediction).
            It is okay for this list to be empty (if no correct subgoal can
            be proposed).
        """
        raise NotImplementedError


TreeStats = collections.namedtuple('TreeStats', [
    'subgoal_sum', 'nodes', 'leaves'
])


class Node:
    """Base class for nodes of the search tree.

    Attrs:
        prior_probability (float or None): Prior probability, if the algorithm
            uses it.
        children (list): List of children, indexed by action.
        count (int): Number of visits in this node.
        is_leaf (bool): Whether the node is a leaf, i.e. has not been expanded
            yet.
    """

    prior_probability = None

    def __init__(self, state):
        self.state = state
        self.children = []
        self.children_infos = []
        self.was_expanded = False

    def visit(self, reward, value, discount):
        """Records a visit in the node during backpropagation.

        Args:
            reward (float or None): Reward obtained when stepping into this
                node, or None if it's the root.
            value (float): Value accumulated on the path out of the node.
            discount (float): Discount factor.
        """
        raise NotImplementedError

    def quality(self, discount):
        """Returns the quality of going into this node in the search tree.

        We use it instead of value, so we can handle dense rewards.
        quality(s, a) = reward(s, a) + discount * value(s')
        """
        raise NotImplementedError

    @property
    def count(self):
        raise NotImplementedError

    def value(self, discount):
        """Returns the value of going into this node in the search tree.

        We use it only to provide targets for value network training.
        value(s) = expected_a quality(s, a)
        """
        return (
            sum(
                child.quality(discount) * child.count for child in self.children
            ) / sum(child.count for child in self.children)
        )

    @property
    def is_leaf(self):
        return not self.children


def find_solving_actions(node):
    if node.is_leaf:
        return None

    for child, child_info in zip(node.children, node.children_infos):
        cur_action_state = (child_info.action_list, child.state)
        if child_info.done:
            return [cur_action_state]
        maybe_actions_states = find_solving_actions(child)
        if maybe_actions_states is not None:
            return [cur_action_state] + maybe_actions_states


def compute_tree_stats_helper(node):
    if node.is_leaf:
        return TreeStats(subgoal_sum=0, nodes=1, leaves=1)

    result = TreeStats(subgoal_sum=len(node.children), nodes=1, leaves=0)
    for child in node.children:
        stats = compute_tree_stats_helper(child)
        result = TreeStats(
            subgoal_sum=result.subgoal_sum + stats.subgoal_sum,
            nodes=result.nodes + stats.nodes,
            leaves=result.leaves + stats.leaves,
        )
    return result


def compute_tree_stats(node):
    stats = compute_tree_stats_helper(node)
    return {
        'nodes': stats.nodes,
        'leaves': stats.leaves,
        'inner_nodes': stats.nodes - stats.leaves,
        'inner_nodes_degree_mean': stats.subgoal_sum / max((stats.nodes - stats.leaves), 1),
    }


class DeadEnd(Exception):
    """Exception raised when no action can be taken during tree traversal."""


class TreeSearchAgent(base.OnlineAgent):
    """Tree search base class."""

    def __init__(
        self,
        n_passes=10,
        discount=0.99,
        depth_limit=float('+inf'),
        n_leaves_to_expand=1,
        model_class=models.PerfectModel,
        keep_tree_between_steps=True,
        callback_classes=None,
        **kwargs
    ):
        """Initializes TreeSearchAgent.

        Args:
            n_passes (int): Number of tree search passes per act().
            discount (float): Discount factor.
            depth_limit (int): Maximum number of nodes visited in a single.
            n_leaves_to_expand (int): Number of consecutive leaves to expand in
                a single pass.
            model_class (type): Subclass of models.EnvModel to create a model.
            keep_tree_between_steps (bool): Whether to keep tree between
                different act() function calls.
            callback_classes (list of types): Classes of callbacks to be called
                by the agent.
            **kwargs: OnlineAgent init keyword arguments.
        """
        if not callback_classes:
            callback_classes = []
        if not model_class.is_perfect:
            callback_classes = [ImperfectModelCallback] + callback_classes
        super().__init__(callback_classes=callback_classes, **kwargs)

        self.n_passes = n_passes
        self._discount = discount
        self._depth_limit = depth_limit
        self._n_leaves_to_expand = n_leaves_to_expand
        self._model_class = model_class
        self._model = None
        self._root = None
        self._root_state = None
        self._keep_tree_between_steps = keep_tree_between_steps

        self._dead_end_hits = None

    @property
    def discount(self):
        return self._discount

    @property
    def model(self):
        return self._model

    def reset_tree(self, state):
        self._root = self._init_root_node(state)

    def _init_root_node(self, state):
        """Initializes the root node of the tree.

        Args:
            state: Model's state corresponding to the root node.

        Returns:
            Node: The initialized root.
        """
        raise NotImplementedError

    def _init_child_nodes(self, leaf, observation):
        """Initializes child nodes for a given leaf.

        Args:
            leaf (Node): Leaf to initialize children for.
            observation (np.ndarray): Observation received at leaf.

        Yields:
            Network prediction requests.

        Returns:
            list: List of Nodes - initialized children.
        """
        raise NotImplementedError

        # To indicate it's a coroutine:
        yield  # pylint: disable=unreachable

    def network_signature(self, observation_space, action_space):
        """Defines the signature of networks used by this TreeSearchAgent.

        Args:
            observation_space (gym.Space): Environment observation space.
            action_space (gym.Space): Environment action space.

        Returns:
            NetworkSignature or None: Either the network signature or None if
            the TreeSearchAgent doesn't use a network.
        """
        raise NotImplementedError

    def _before_pass(self):
        """Called before each pass."""

    def _before_model_step(self, node):
        """Called before each step on the model.

        Specifically, before any computation is performed to determine the
        action to take on the model.

        Args:
            node (Node): Outgoing node.
        """

    def _before_real_step(self, node):
        """Called before each step on the real environment.

        Specifically, after all passes but before the action to take on the
        real environment is determined.

        Args:
            node (Node): Outgoing node.
        """

    def _on_new_root(self, root):
        """Called when the root node changes.

        But only if it's already expanded - if it's not, it gets called right
        after expansion.

        Args:
            root (Node): New root.
        """

    def _make_filter_fn(self, exploratory):
        """Creates a filter function for selecting child nodes.

        Args:
            exploratory (bool): Whether the choice of the child node is
                exploratory (during tree traversal) or not (when choosing the
                final action on the real environment).

        Returns:
            callable: Function Node -> bool indicating if a child should be
                included in selection.
        """
        del exploratory
        return lambda _: True

    @property
    def _zero_quality(self):
        """Quality backpropagated in case of a "done"."""
        return 0

    @property
    def _dead_end_quality(self):
        """Quality backpropagated in case of a "dead end"."""
        return 0

    def _choose_child(self, node, exploratory, strict_filter=True):
        """Chooses a child of a node during tree traversal or on the real env.

        Wrapper around _choose_action, handling node filtering and dead ends.

        Args:
            node (Node): Node to choose an action from.
            exploratory (bool): Whether the choice should be exploratory (during
                tree traversal) or not (when choosing the final action on the
                real environment).
            strict_filter (bool): In case all actions were filtered-out:
                True: throw DeadEnd exception
                False: ignore the filter and choose from all the actions.

        Raises:
            DeadEnd: When no action can be taken.

        Returns:
            Pair (child, action leading to it).
        """
        filter_fn = self._make_filter_fn(exploratory)
        actions = [
            action for (action, child) in enumerate(node.children)
            if filter_fn(child)
        ]
        if not actions:
            if strict_filter:
                raise DeadEnd
            # All actions were filtered-out. Ignore the filter.
            actions = list(range(len(node.children)))

        action = self._choose_action(
            node, actions, exploratory=exploratory
        )
        assert action in actions, (
            'Invalid implementation of _choose_action: action '
            '{} disallowed.'.format(action)
        )
        return (node.children[action], action)

    def _choose_action(self, node, actions, exploratory):
        """Chooses the action to take in a given node.

        Args:
            node (Node): Node to choose an action from.
            actions (list): List of allowed actions.
            exploratory (bool): Whether the choice should be exploratory (during
                tree traversal) or not (when choosing the final action on the
                real environment).

        Returns:
            Action to take.
        """
        raise NotImplementedError

    def _traverse(self, root, observation, path):
        """Chooses a path from the root to a leaf in the search tree.

        By default, traverses the tree top-down, calling _choose_child in each
        node. Can be overridden in derived classes to customize this behavior.

        Args:
            root (Node): Root of the search tree.
            observation (np.ndarray): Observation received at root.
            path (list): Empty list to populate with pairs (reward, node)
                encountered during traversal. We get it as an argument, so
                we can recover it in case a DeadEnd is raised.

        Yields:
            Network prediction requests.

        Returns:
            Pair (path, quality), where path is the path passed as an argument,
            and quality is the quality of the last node on the path.
        """
        assert not path
        # (reward, node)
        path.append((None, root))
        node = root
        done = False
        n_leaves_left = self._n_leaves_to_expand
        quality = self._zero_quality
        while not node.is_leaf and not done and len(path) < self._depth_limit:
            agent_info = self._compute_node_info(node)

            self._before_model_step(node)
            parent = node
            (node, action) = self._choose_child(
                node, exploratory=True, strict_filter=True
            )
            (observation, reward, done) = (
                parent.children[action].state,
                parent.children_infos[action].reward,
                parent.children_infos[action].done,
            )

            for callback in self._callbacks:
                callback.on_model_step(
                    agent_info, action, observation, reward, done
                )

            path.append((reward, node))

            if node.is_leaf and n_leaves_left > 0:
                if not done:
                    quality = self._expand_leaf(node, observation)
                n_leaves_left -= 1

        if node.is_leaf and not done and n_leaves_left > 0:
            # Corner case: we've reached the depth limit and we're at the leaf.
            # We need to expand it.
            quality = self._expand_leaf(node, observation)

        return (path, quality)

    def _expand_leaf(self, leaf, observation):
        """Expands a leaf and returns its quality.

        The leaf's new children are assigned initial quality. The quality of the
        "best" new leaf is then backpropagated.

        Only modifies leaf - adds children with new qualities.

        Args:
            leaf (Node): Leaf to expand.
            observation (np.ndarray): Observation received at leaf.

        Yields:
            Network prediction requests.

        Returns:
            float: Quality of a chosen child of the expanded leaf.
        """
        if leaf.was_expanded:
            # Spare some computation. If this node was tried to be expanded,
            # but it is still a leaf, it doesn't make sense to try that again.
            self._dead_end_hits += 1
            raise DeadEnd
        leaf.was_expanded = True

        (leaf.children, leaf.children_infos) = \
            self._init_child_nodes(leaf, observation)
        for node in leaf.children:
            quality = node.quality(self._discount)
            prob = node.prior_probability
            prob_ok = prob is None or np.isscalar(prob)
            assert np.isscalar(quality) and prob_ok, (
                'Invalid shape of node quality or prior probability - expected '
                'scalars, got {} and {}. Check if your network architecture is '
                'appropriate for the observation shape.'.format(
                    quality.shape, prob.shape if prob is not None else None
                )
            )

        if leaf is self._root:
            self._on_new_root(leaf)

        (child, _) = self._choose_child(
            leaf, exploratory=True, strict_filter=True
        )
        return child.quality(self._discount)

    def _backpropagate(self, quality, path):
        """Backpropagates quality to the root through path.

        Only modifies the qualities of nodes on the path.

        Args:
            quality (float): Quality collected at the leaf.
            path (list): List of (reward, node) pairs, describing a path from
                the root to a leaf.
        """
        for (reward, node) in reversed(path):
            node.visit(reward, value=quality, discount=self._discount)
            if reward is None:
                break
            quality = reward + self._discount * quality

    def _run_pass(self, root, observation):
        """Runs a pass of tree search.

        A pass consists of:
            1. Tree traversal to find a leaf, or until reaching the depth limit.
            2. Expansion of the leaf, or several consecutive leaves, adding
               their successor states to the tree and rating them.
            3. Backpropagation of the value of the best child of the last node
               on the path.

        Args:
            root (Node): Root node.
            observation (np.ndarray): Observation collected at the root.

        Yields:
            Network prediction requests.
        """
        for callback in self._callbacks:
            callback.on_pass_begin()

        path = []
        try:
            (path, quality) = self._traverse(root, observation, path)
        except DeadEnd:
            quality = self._dead_end_quality
        self._backpropagate(quality, path)

        for callback in self._callbacks:
            callback.on_pass_end()

    def reset(self, observation):
        """Reinitializes the search tree for a new environment."""
        self._root = self._init_root_node(observation)

    def act(self, observation):
        """Runs n_passes tree search passes and chooses the best action."""
        self._root_state = observation

        if not self._keep_tree_between_steps:
            self._root = self._init_root_node(self._root_state)

        self._dead_end_hits = 0
        for _ in range(self.n_passes):
            self._before_pass()
            self._run_pass(self._root, observation)
        if len(self._root.children) == 0:
            return None

        agent_info = {
            '_node': self._root,
            'dead_end_hits': self._dead_end_hits,
        }
        agent_info.update(self._compute_node_info(self._root))
        agent_info.update(self._compute_tree_metrics(self._root))

        self._before_real_step(self._root)
        (new_root, action) = self._choose_child(
            self._root, exploratory=False, strict_filter=False
        )
        multi_step_info = self._root.children_infos[action]

        self._root = new_root
        if not self._root.is_leaf:
            self._on_new_root(self._root)

        return action, new_root.state, multi_step_info, agent_info

    def postprocess_transitions(self, transitions):
        for transition in transitions:
            transition.agent_info.update(
                self._compute_node_info(transition.agent_info.pop('_node'))
            )
        return transitions

    def _compute_node_info(self, node):
        value = node.value(self._discount)
        qualities = np.array(
            [child.quality(self._discount) for child in node.children]
        )
        action_counts = np.array([child.count for child in node.children])
        # "Smooth" histogram takes into account the initial actions
        # performed on all children of an expanded leaf, resulting in
        # a more spread out distribution.
        action_histogram_smooth = action_counts / (np.sum(action_counts) + 1e-6)
        # Ordinary histogram only takes into account the actual actions
        # chosen in the inner nodes.
        action_histogram = (action_counts - 1) / (
            np.sum(action_counts - 1) + 1e-6
        )
        return {
            'value': value,
            'qualities': qualities,
            'action_histogram_smooth': action_histogram_smooth,
            'action_histogram': action_histogram,
        }

    def _compute_tree_metrics(self, root):
        leaf_depths = []
        path = [root]
        # Number of visited children for a given node in the path.
        children_visited = [0]

        def go_to_parent():
            path.pop()
            children_visited.pop()

        # Iterate over leaves with DFS.
        while path:
            node = path[-1]
            if node.is_leaf:
                leaf_depths.append(len(path) - 1)
                go_to_parent()
            elif children_visited[-1] == len(node.children):
                # All children of the given node were already visited.
                go_to_parent()
            else:
                # Expand the new child.
                path.append(node.children[children_visited[-1]])
                children_visited[-1] += 1
                children_visited.append(0)

        return {
            'depth_mean': sum(leaf_depths) / len(leaf_depths),
            'depth_max': max(leaf_depths),
        }

    def _compute_model_prediction(self, action):
        prediction = yield from self._model.predict_steps(
            [action], include_state=False
        )
        keys = ['predicted_observation', 'predicted_reward', 'predicted_done']
        return {
            key: value
            for key, [value] in zip(keys, prediction)
        }

    @staticmethod
    def compute_metrics(episodes):
        def episode_info(key):
            for episode in episodes:
                yield from episode.transition_batch.agent_info[key]

        def entropy(probs):
            def plogp(p):
                # If out this case to avoid log(0).
                return p * np.log(p) if p else 0
            return -np.sum([plogp(p) for p in probs])

        return {
            'depth_mean': np.mean(list(episode_info('depth_mean'))),
            'depth_max': max(episode_info('depth_max')),
            'entropy_mean': np.mean(
                list(map(entropy, episode_info('action_histogram')))
            ),
            'entropy_smooth_mean': np.mean(
                list(map(entropy, episode_info('action_histogram_smooth')))
            ),
            **metric_utils.compute_scalar_statistics(
                list(episode_info('value')),
                prefix='value',
                with_min_and_max=True
            ),
        }


class ImperfectModelCallback(base.AgentCallback):
    """Callback which reacts on potential model's mispredictions."""
    def __init__(self, agent):
        super().__init__(agent)
        self._last_observation = None

    def on_episode_begin(self, env, observation, epoch):
        self._last_observation = observation

    def on_real_step(self, agent_info, action, observation, reward, done):
        if not ops.nested_array_equal(
                agent_info['predicted_observation'],
                observation
        ):
            # We do catch-up only to get the correct_state. Model would pick up
            # the correct state at the beginning of TreeSearchAgent.act()
            # method anyway.
            self._agent.model.catch_up(observation)
            correct_state = self._agent.model.clone_state()
            # Reset the tree, because it has been built on a wrong state.
            self._agent.reset_tree(correct_state)

        # Feed the model with a gold-standard transition from the real env.
        # We do that even if the model's prediciton was right.
        self._agent.model.correct(
            self._last_observation, action, observation, reward, done,
            agent_info
        )

        self._last_observation = observation
