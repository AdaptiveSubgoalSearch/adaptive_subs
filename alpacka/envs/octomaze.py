"""Octomaze toy problem.

Created to benchmark Shoot Tree Search.
"""

import numpy as np
from gym import spaces

from alpacka.envs import base


class Octomaze(base.RestorableEnv):
    """Octomaze toy problem.

    It's characterized by one initial state that connects with the only terminal
    state through many long corridors.
    """

    stochasticity = base.Stochasticity.none

    class State:
        """Octomaze state data class.

        Valid states:
          - Initial state when corridor is None.
          - Terminal state when terminal is True.
          - Intermediate state described by corridor and steps otherwise.
        """

        def __init__(self, corridor, steps, terminal=False):
            self.corridor = corridor
            self.steps = steps
            self.terminal = terminal
            self._hash = None

        # pylint: disable=redefined-builtin
        def __hash__(self):
            if self._hash is None:
                if self.corridor is None: # Initial state hash.
                    self._hash = -1
                elif self.terminal:
                    # Terminal state hash.
                    self._hash = 1
                else:
                    self._hash = hash((self.corridor, self.steps))
            return self._hash

        def __eq__(self, other):
            if self.corridor is None and other.corridor is None:
                return True
            elif self.terminal == other.terminal:
                return True
            elif self.corridor == other.corridor and self.steps == other.steps:
                return True
            else:
                return False

    def __init__(self, num_corridors=8, corridor_length=10):
        """Initializes Octomaze.

        Args:
            num_corridors (int): Number of corridors connecting the initial
                state with the terminal state.
            corridor_length (int): Length of each corridor.
        """
        self._num_corridors = num_corridors
        self._corridor_length = corridor_length
        self._state = None

        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self._num_corridors, self._corridor_length),
            dtype=np.float32)
        self.action_space = spaces.Discrete(self._num_corridors)

    def reset(self):
        self._state = Octomaze.State(corridor=None, steps=None)
        return self._get_transition()[0]

    def step(self, action):
        if self._state.corridor is None:
            # Pick a corridor to go through.
            self._state.corridor = action
            self._state.steps = 0
        elif self._state.terminal:
            raise ValueError('Step after done.')
        elif self._state.corridor == action:
            # Go one step further in the chosen corridor.
            self._state.steps += 1

            # Check if stepped into the terminal state.
            if self._state.steps == self._corridor_length:
                self._state.terminal = True

        return self._get_transition()

    def render(self, mode='human'):
        raise NotImplementedError

    def clone_state(self):
        """Returns the current environment state."""
        return Octomaze.State(corridor=self._state.corridor,
                              steps=self._state.steps,
                              terminal=self._state.terminal)

    def restore_state(self, state):
        """Restores environment state, returns the observation."""
        self._state = Octomaze.State(corridor=state.corridor,
                                     steps=state.steps,
                                     terminal=state.terminal)
        return self._get_transition()[0]

    @property
    def valid_actions(self):
        """Returns iterable collection of valid actions in the current state."""
        if self._state.corridor is None:
            # Can pick any corridor to go through in the initial state.
            return range(self.action_space.n)
        elif self._state.terminal:
            # No valid actions after done.
            return []
        else:
            # Can only go further in the chosen corridor.
            return [self._state.corridor]

    def _get_transition(self):
        if self._state.corridor is None:
            # Initial state.
            return (np.zeros(self.observation_space.shape),)
        elif self._state.terminal:
            # Terminal state.
            observation = np.ones(self.observation_space.shape)
            return (observation, 1., True, {'solved': True})
        else:
            # Intermediate state.
            observation = np.zeros(self.observation_space.shape)
            observation[self._state.corridor][self._state.steps] = 1
            return (observation, 0., False, {'solved': False})
