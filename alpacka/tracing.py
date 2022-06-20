"""Debug tracing of agent's decisions.

This module allows tracing of the decisions made by both model-free and
model-based agents. For the latter, the traces contain the trajectory performed
on the real environment as well as the planning passes made when choosing
actions.

Traces are dumped to disk and can be viewed using our trace explorer TraceX -
see the README in alpacka/tools/tracex.

Information collected in traces:
    state_info: Either the value of the attribute state_info of the environment
        if it's defined, or the current observation otherwise. State info can be
        visualized in TraceX. To support that, define a nested class Renderer
        subclassing envs.EnvRenderer in your Env.
    action: Action from the environment. Can be shown in a human-readable form.
        To support that, define a nested class Renderer subclassing
        envs.EnvRenderer in your Env.
    reward: Reward from the environment.
    terminal: Done flag from the environment.
    agent_info: The info dict returned in OnlineAgent.act(). Keys should be
        string. Values can be either scalar or vector. Vector values are assumed
        to be of length equal to the number of actions and visualized in TraceX
        as histograms.
"""


import collections
import copy
import lzma
import os
import pickle
import random

import gin

from alpacka import agents
from alpacka import envs


# Data structures for traces.
# ===========================


# Trace of an agent's planning algorithm.
Trace = collections.namedtuple('Trace', [
    'renderer',    # EnvRenderer
    'trajectory',  # Trajectory
    'tree',        # [TreeNode], indexed by node_id
])


Trajectory = collections.namedtuple('Trajectory', [
    'init_state',   # State
    'transitions',  # [Transition]
])


# State of the environment.
State = collections.namedtuple('State', [
    'state_info',   # Env-specific
    'node_id',      # int
    'terminal',     # bool
])


transition_fields = [
    'agent_info',  # dict, Agent-specific, from previous state
    'action',      # Env-specific
    'reward',      # float
    'to_state',    # State
]


# Transition in the real environment.
RealTransition = collections.namedtuple('RealTransition', transition_fields + [
    'passes',      # [[ModelTransition]], from previous state
])


# Transition in the model.
ModelTransition = collections.namedtuple('ModelTransition', transition_fields)


# Node of the planning tree.
TreeNode = collections.namedtuple('TreeNode', [
    'agent_info',  # Agent-specific
    'children',    # dict action -> node_id
])


class DummyRenderer(envs.EnvRenderer):

    def render_state(self, state_info):
        """Renders state_info to an image."""
        del state_info

    def render_action(self, action):
        """Renders action to a string."""
        return str(action)


@gin.configurable
class TraceCallback(agents.AgentCallback):
    """Callback for collecting traces."""

    def __init__(self, agent, output_dir=None, sample_rate=1.0):
        """Initializes TraceCallback.

        Args:
            agent (Agent): Agent which owns this callback.
            output_dir (str): Directory to save traces in, or None if traces
                shouldn't be saved.
            sample_rate (float): Fraction of episodes to trace.
        """
        super().__init__(agent)
        self._output_dir = output_dir
        if self._output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        self._sample_rate = sample_rate

        self._env = None
        self._epoch = None
        self._trace = None
        # Trajectory trace.
        self._trajectory = None
        self._passes = None
        self._pass = None
        # Tree trace.
        self._tree_nodes = None
        self._current_root_id = None
        self._current_node_id = None

    @property
    def trace(self):
        return self._trace

    def on_episode_begin(self, env, observation, epoch):
        """Called in the beginning of a new episode."""
        # Sample a fraction of episodes to dump traces from.
        if random.random() > self._sample_rate:
            return

        self._env = env
        self._epoch = epoch
        self._trace = None
        state_info = getattr(self._env, 'state_info', observation)
        self._trajectory = Trajectory(
            init_state=State(
                state_info=state_info,
                node_id=0,
                terminal=False,
            ),
            transitions=[],
        )
        self._passes = []

        self._tree_nodes = [TreeNode(agent_info=None, children={})]
        self._current_root_id = 0

    def on_pass_begin(self):
        """Called in the beginning of every planning pass."""
        if self._trajectory is None:
            return

        self._pass = []
        self._current_node_id = self._current_root_id

    def on_model_step(self, agent_info, action, observation, reward, done):
        """Called after every step in the model."""
        if self._trajectory is None:
            return

        self._current_node_id = self._traverse_tree(
            self._current_node_id, agent_info, action
        )

        state_info = getattr(self._env, 'state_info', observation)
        self._pass.append(copy.deepcopy(ModelTransition(
            agent_info=self._filter_agent_info(agent_info),
            action=action,
            reward=reward,
            to_state=State(
                state_info=state_info,
                node_id=self._current_node_id,
                terminal=done,
            ),
        )))

    def on_pass_end(self):
        """Called in the end of every planning pass."""
        if self._trajectory is None:
            return

        self._passes.append(self._pass)
        self._pass = None

    def on_real_step(self, agent_info, action, observation, reward, done):
        """Called after every step in the real environment."""
        if self._trajectory is None:
            return

        self._current_root_id = self._traverse_tree(
            self._current_root_id, agent_info, action
        )
        self._current_node_id = self._current_root_id

        state_info = getattr(self._env, 'state_info', observation)
        self._trajectory.transitions.append(copy.deepcopy(RealTransition(
            agent_info=self._filter_agent_info(agent_info),
            passes=self._passes,
            action=action,
            reward=reward,
            to_state=State(
                state_info=state_info,
                node_id=self._current_root_id,
                terminal=done,
            ),
        )))
        self._passes = []

    def on_episode_end(self):
        """Called in the end of an episode."""
        if self._trajectory is None:
            return

        try:
            renderer_class = type(self._env.unwrapped).Renderer
        except AttributeError:
            renderer_class = DummyRenderer

        self._trace = Trace(
            renderer=renderer_class(self._env),
            trajectory=self._trajectory,
            tree=self._tree_nodes,
        )
        self._env = None
        self._trajectory = None
        self._passes = None
        self._pass = None
        self._tree_nodes = None
        self._current_root_id = None
        self._current_node_id = None

        if self._output_dir is not None:
            # Create a random, hopefully unique hexadecimal id for the trace to
            # avoid race conditions.
            trace_path = os.path.join(
                self._output_dir, 'ep{}_{:06x}'.format(
                    self._epoch, random.randrange(16 ** 6)
                )
            )
            dump(self.trace, trace_path)

    @property
    def _next_node_id(self):
        return len(self._tree_nodes)

    def _traverse_tree(self, node_id, agent_info, action):
        node = self._tree_nodes[node_id]._replace(
            agent_info=self._filter_agent_info(agent_info)
        )
        self._tree_nodes[node_id] = node
        if action not in node.children:
            node.children[action] = self._next_node_id
            self._tree_nodes.append(TreeNode(
                agent_info=None,
                children={},
            ))
        return node.children[action]

    @staticmethod
    def _filter_agent_info(agent_info):
        """Filters out "private" keys - starting with an underscore."""
        return {
            key: value
            for (key, value) in agent_info.items()
            if not key.startswith('_')
        }


def dump(trace, path):
    """Dumps a trace to a given path."""
    with lzma.open(path, 'wb') as f:
        pickle.dump(trace, f)


def load(path):
    """Loads a trace from a given path."""
    with lzma.open(path, 'rb') as f:
        return pickle.load(f)
