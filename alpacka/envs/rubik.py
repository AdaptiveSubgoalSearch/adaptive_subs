"""Rubik's cube environment."""

import math

import gin
import numpy as np
from gym import spaces
from scipy.spatial import transform

from alpacka.envs import base


def rotate(v, along, direction=1):
    return tuple(transform.Rotation.from_rotvec(
        np.multiply(along, direction * np.pi / 2)).apply(v))


def get_permutation(stickers, moved_stickers):
    """Compute the permutation which turns stickers to moved_stickers."""
    permutation = [0] * len(stickers)

    for i, moved_sticker in enumerate(moved_stickers):
        found = False
        # Find the rotated sticker on the initial list.
        for j, sticker in enumerate(stickers):
            if np.allclose(moved_sticker[0], sticker[0]):
                permutation[i] = j
                found = True
                break

        assert found

    return np.array(permutation)


def print_coloured_square(colour):
    """Prints a coloured square on stdout."""
    if colour == -1:  # none
        pass
    elif colour == 0:  # white
        print('\x1b[1;107m', end='')
    elif colour == 1:  # yellow
        print('\x1b[1;103m', end='')
    elif colour == 2:  # orange
        print('\x1b[1;44m', end='')
    elif colour == 3:  # blue
        print('\x1b[1;42m', end='')
    elif colour == 4:  # red
        print('\x1b[0;43m', end='')
    elif colour == 5:  # green
        print('\x1b[1;41m', end='')
    else:
        assert False, 'Colour must be in range [-1, 5]'

    print('  \x1b[1;0m', end='')


# Default reward functions
@gin.configurable
def penalize_step(state, goal):
    return 0 if np.array_equal(state, goal) else -1


@gin.configurable
def on_reached(state, goal):
    return 1 if np.array_equal(state, goal) else 0


@gin.configurable
def matching_stickers(state, goal):
    return np.average(state == goal)


class RubiksCube(base.RestorableEnv):
    """Rubik's n-cube environment."""

    stochasticity = base.Stochasticity.episodic

    def __init__(self, n=3, n_shuffles=100, step_limit=100,
                 reward_func=penalize_step, colour_pattern=None,
                 smart_shuffles=True):
        """
        :param n: Size of the cube. Default n=3 denotes the standard 3x3x3 cube.
        :param n_shuffles: The exact number of random moves applied every time
        to mix the cube.
        :param step_limit: Maximal length of a single episode.
        :param reward_func: The reward function used in the environment. Can be
        either a a pre-implemented function (see default reward functions above)
        or a custom function object of type (state, goal) -> float.
        :param colour_pattern: List of colours for all the faces. This can be
        used to simplify the problem by identifying some colours. The list must
        be of length 6 and must contain integers from range [0,5]. Standard
        pattern [0,1,2,3,4,5] denotes that all the faces have different colours.
        If some values on this list are equal, the matching faces contain
        indistinguishable stickers. For example, pattern [0,0,1,1,2,2]
        identifies opposite faces' colours. Faces on the list are listed in
        order [U,D,F,B,R,L]. The size of observations is adjusted to the values
        provided in the pattern.
        :param smart_shuffles: Whether to use simple heuristics to avoid
        redundant moves during shuffling.
        """
        self._n = n

        self.n_shuffles = n_shuffles
        self.step_limit = step_limit
        self.compute_reward = reward_func
        self.smart_shuffles = smart_shuffles

        self._colour_pattern = np.arange(
            6) if colour_pattern is None else np.array(colour_pattern).astype(
            np.int)
        self._n_colours = None
        self._check_colour_pattern()

        self._state = None
        self._moves = []
        self._steps = 0

        self._observation_shape = (6, n, n, self.n_colours)
        self.action_space = spaces.Discrete(6 * (n // 2) * 2)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=self._observation_shape,
                                            dtype=np.float32)

        self._setup_state()
        self._setup_moves()

        self._goal_state = self._state
        self._shaped_goal_state = self._get_state()

    def _check_colour_pattern(self):
        """Verifies whether given colour pattern meets the requirements."""
        assert len(self._colour_pattern) == 6 and (
                self._colour_pattern >= 0).all() and (
                       self._colour_pattern <= 5).all()
        self.n_colours = len(set(self._colour_pattern))

    def _setup_state(self):
        """Brings the state to solved form."""
        state = np.arange(6 * self._n * self._n) // (self._n * self._n)
        self._state = self._colour_pattern[state]

    def _setup_moves(self):
        """Computes the permutations associated with all the moves.

        Every state of the Rubik's cube is defined by the order of stickers on
        faces. Every move, i.e. rotating a face, corresponds to a fixed
        permutation applied to the current state. This function computes the
        desired permutations and fixes them as parameters, so it needs to be
        done only once.

        Firstly, associate every sticker with a vector pointing to its position.
        Then, for every possible move, which is in fact a rotation of some
        vectors, apply the transformation and observe the permutation obtained
        this way.
        """

        faces = [0, 1, 2, 3, 4, 5]
        centers = [(0, 1, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1), (1, 0, 0),
                   (-1, 0, 0)]
        # Fix an orientation of stickers on every face.
        directions = [[(-1, 0, 0), (0, 0, 1)], [(-1, 0, 0), (0, 0, -1)],
                      [(-1, 0, 0), (0, 1, 0)],
                      [(1, 0, 0), (0, 1, 0)], [(0, 0, -1), (0, 1, 0)],
                      [(0, 0, 1), (0, 1, 0)]]

        # Container for vectors pointing to all the stickers.
        stickers = []

        for f in faces:
            for i in range(self._n):
                for j in range(self._n):
                    # Determine the vector of sticker on the f-th face, on
                    # position (i,j) in its NxN matrix.
                    move_y = np.multiply(directions[f][0],
                                         1 - (2 * i + 1) / self._n)
                    move_x = np.multiply(directions[f][1],
                                         (2 * j + 1) / self._n - 1)

                    stickers.append(
                        [np.add(centers[f], np.add(move_x, move_y)), f])

        for f in faces:  # face which determines the rotation axis
            for level in range(self._n // 2):  # which layer will be rotated
                for direction in [-1, 1]:  # either clockwise or not
                    # Compute the coordinate of the rotating layer.
                    dist = 1 - (2 * level + 1) / self._n
                    moved_stickers = []

                    # Rotate vectors on the selected layer.
                    for s in stickers:
                        if math.isclose(dist, np.dot(centers[f], s[0])) or (
                                level == 0 and
                                math.isclose(1, np.dot(centers[f], s[0]))):
                            moved_stickers.append(
                                [rotate(s[0], centers[f], direction), s[1]])
                        else:
                            moved_stickers.append(s)

                    self._moves.append(
                        get_permutation(stickers, moved_stickers))

    def reset(self):
        self._setup_state()
        self._shuffle()

        self._steps = 0

        return self._get_state()

    def step(self, action):
        self._move_by_action(action)
        reward = self.compute_reward(self._state, self._goal_state)

        self._steps += 1
        done = np.array_equal(self._state, self._goal_state) or \
               self._steps >= self.step_limit

        return self._get_state(), reward, done, dict()

    def render(self, mode='human'):
        figure = np.full((3 * self._n, 4 * self._n), -1)
        state = np.reshape(self._state, self._observation_shape[:-1])
        order = [0, 5, 2, 4, 3, 1]

        figure[0:self._n, self._n:2 * self._n] = np.rot90(state[order[0]])

        for i in range(4):
            face = np.rot90(state[order[i + 1]], 1)
            figure[self._n:2 * self._n, i * self._n:(i + 1) * self._n] = face

        face = np.rot90(state[order[5]])
        figure[2 * self._n:3 * self._n, self._n:2 * self._n] = face

        for i in range(figure.shape[0]):
            for j in range(figure.shape[1]):
                print_coloured_square(figure[i, j])
            print()

    def clone_state(self):
        return tuple(self._state), self._steps

    def restore_state(self, state):
        self._state, self._steps = state
        self._state = np.array(self._state)
        return self._get_state()

    def get_goal_state(self):
        return self._shaped_goal_state

    @staticmethod
    def reverse_action(action):
        # Actions are grouped in complementary pairs.
        return action - (action % 2) + (1 - (action % 2))

    def _move_by_action(self, action):
        """Performs a move determined by the given action."""
        self._state = self._state[self._moves[action]]

    def _is_reversed(self, action1, action2):
        """Checks whether the two given actions are mutually reverse."""
        if action1 is None or action2 is None:
            return False
        if action1 == action2:
            return False
        return action1 // 2 == action2 // 2

    def _shuffle(self):
        """Shuffles the cube."""
        last_action = None
        last_action_counter = 0

        for _ in range(self.n_shuffles):
            while True:
                action = self.action_space.sample()
                if not self.smart_shuffles:
                    break
                if self._is_reversed(action, last_action):
                    continue
                if action == last_action and last_action_counter >= 3:
                    continue
                break

            if last_action == action:
                last_action_counter += 1
            else:
                last_action = action
                last_action_counter = 1
            self._move_by_action(action)

    def _get_state(self):
        """Returns current state of the cube, formatted."""
        return np.reshape(np.eye(self.n_colours)[self._state],
                          self._observation_shape)
