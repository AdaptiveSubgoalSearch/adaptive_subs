"""PyColab gridworld environments."""

import curses
import pickle

import gym
import gym_pycolab
import numpy as np
import pycolab
from matplotlib import colors
from matplotlib import pyplot as plt
from pycolab.examples.classics import chain_walk
from pycolab.examples.classics import cliff_walk
from pycolab.examples.classics import four_rooms

from alpacka.envs import base


class PyColabEnv(base.RestorableEnv, gym_pycolab.PyColabEnv):
    """Base class for envs based on PyColab."""

    allowed_stochasticities = None

    def __init__(
        self,
        n_actions,
        max_iterations=200,
        default_reward=0,
        solved_at_reward=1,
        hidden_characters=None,
        stochasticity=base.Stochasticity.unknown,
        init_seed=0,
    ):
        # Save the seed before super().__init__ in case it wants to do some
        # random operations.
        self._init_seed = init_seed

        super().__init__(
            max_iterations=max_iterations,
            default_reward=default_reward,
            action_space=gym.spaces.Discrete(n_actions),
            resize_scale=8,
        )
        self._solved_at_reward = solved_at_reward
        self._hidden_characters = hidden_characters or set()
        if self.allowed_stochasticities is not None:
            assert stochasticity in self.allowed_stochasticities
        self.stochasticity = stochasticity

        self.seed(init_seed)

    def _paint_board(self, layers):
        for c in self._hidden_characters:
            layers[c].fill(False)
        return super()._paint_board(layers)

    def reset(self):
        if self.stochasticity is base.Stochasticity.none:
            # In deterministic mode, re-seed the environment at the beginning
            # of each episode.
            self.seed(self._init_seed)

        return super().reset()

    def clone_state(self):
        # self.current_game is None when the game has ended.
        if self.current_game is not None:
            palette = self.current_game.backdrop.palette
            # Workaround for the fact that the PyColab Palette is not picklable.
            self.current_game.backdrop._p_a_l_e_t_t_e = None  # pylint: disable=protected-access

        state = (
            self.current_game,
            self._last_observations,
            self._last_reward,
        )

        if self.stochasticity is not base.Stochasticity.universal:
            # In non-universal modes, include the rng in state, so random
            # decisions taken twice from the same state are identical.
            state += (self.np_random,)

        # PyColab games don't have a well-defined state, and the game itself is
        # not hashable, so just pickle the whole thing. It's ~1kB.
        state = pickle.dumps(state)
        if self.current_game is not None:
            self.current_game.backdrop._p_a_l_e_t_t_e = palette  # pylint: disable=protected-access

        return state

    def restore_state(self, state):
        if self.current_game is None:
            self.current_game = self.make_game()
        palette = self.current_game.backdrop.palette

        state = pickle.loads(state)
        (self.current_game, observations, reward) = state[:3]

        if self.stochasticity is not base.Stochasticity.universal:
            # In non-universal modes, restore the rng from state.
            self.np_random = state[3]

        if self.current_game is None:
            # Corner case: restored state corresponds to a finished game.
            self.current_game = self.make_game()
            self.current_game._game_over = True  # pylint: disable=protected-access
        self.current_game._backdrop._p_a_l_e_t_t_e = palette  # pylint: disable=protected-access

        self._update_for_game_step(observations, reward)
        return self._last_state

    def step(self, action):
        (observation, reward, done, info) = super().step(action)
        if self._solved_at_reward is not None:
            info['solved'] = reward >= self._solved_at_reward
        return (observation, reward, done, info)

    def make_game(self):
        raise NotImplementedError

    def make_colors(self):
        return {
            '#': (153, 51, 0),
            'P': (255, 0, 0),
            'X': (0, 255, 0),
        }

    def play(self, *args, **kwargs):
        """Default play() for the simplest envs.

        Binds the four arrow keys to actions 0-3.

        Uses ncurses for in-terminal rendering.
        """
        del args, kwargs

        game = self.make_game()
        ui = pycolab.human_ui.CursesUi(
            keys_to_actions={
                curses.KEY_RIGHT: 0,
                curses.KEY_UP: 1,
                curses.KEY_LEFT: 2,
                curses.KEY_DOWN: 3,
            },
            repainter=pycolab.rendering.ObservationCharacterRepainter({
                c: ' ' for c in self._hidden_characters
            }),
            delay=200,  # ms
        )
        ui.play(game)

    @property
    def state_info(self):
        board = self._last_observations.board
        return tuple(map(tuple, board))

    class Renderer(base.EnvRenderer):
        """Default renderer for PyColab envs."""

        def __init__(self, env):
            super().__init__(env)
            self._colors = env.make_colors()

        def render_state(self, state_info):
            def render_tile(idx):
                return self._colors.get(chr(idx), (0, 0, 0))

            render_row = lambda row: list(map(render_tile, row))
            return np.array(
                list(map(render_row, state_info)),
                dtype=np.uint8,
            )

        def render_heatmap(self, heatmap, current_state_info):
            cmap = plt.cm.get_cmap('plasma')
            values = list(heatmap.values()) + [0]
            norm = colors.Normalize(vmin=min(values), vmax=max(values))
            img = self.render_state(current_state_info)
            free_mask = (
                (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)
            )
            for (state_info, value) in heatmap.items():
                player_mask = np.array(state_info) == ord('P')
                color = np.array(cmap(norm(value))[:3]) * 255
                img[free_mask & player_mask] = color
            return img

        def render_action(self, action):
            directions = {
                0: '>',
                1: '^',
                2: '<',
                3: 'v',
            }
            return directions.pop(action, str(action))


def make_pycolab_example_env(name, module, n_actions):
    """Creates an env class based on a game from PyColab examples.

    Args:
        name (str): Name for the created env class.
        module (types.ModuleType): Module under pycolab.examples.
        n_actions (int): Number of actions in the environment.

    Returns:
        The created environment class.
    """
    assert hasattr(module, 'make_game') and hasattr(module, 'main'), (
        'The example module must implement make_game() and main(argv).'
    )

    class Env(PyColabEnv):
        """The PyColab env for the given example game."""

        def __init__(self, *args, **kwargs):
            super().__init__(
                n_actions,
                *args,
                stochasticity=base.Stochasticity.none,
                **kwargs
            )

        def make_game(self):
            return module.make_game()  # pylint: disable=no-member

        def play(self, *args, **kwargs):
            del kwargs
            module.main(argv=args)  # pylint: disable=no-member

    Env.__name__ = name
    return Env


ChainWalk = make_pycolab_example_env('ChainWalk', chain_walk, n_actions=3)
CliffWalk = make_pycolab_example_env('CliffWalk', cliff_walk, n_actions=5)
FourRooms = make_pycolab_example_env('FourRooms', four_rooms, n_actions=5)


class PlayerSprite(pycolab.prefab_parts.sprites.MazeWalker):
    """Basic sprite for a player walking in 4 directions.

    Can't pass through walls, denoted by # and -.
    Gets step_reward after each step.
    """

    def __init__(self, corner, position, character, step_reward=0):
        super().__init__(corner, position, character, impassable='#-')
        self._step_reward = step_reward

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del layers, backdrop, things

        directions = {
            0: self._east,
            1: self._north,
            2: self._west,
            3: self._south,
        }
        if actions in directions:
            directions[actions](board, the_plot)

        the_plot.add_reward(self._step_reward)


class GoalSprite(pycolab.things.Sprite):
    """Sprite for the goal position.

    When it's reached by the player, gives a reward and terminates the episode.
    """

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del actions, board, layers, backdrop
        # Is the player on the goal?
        if things['P'].position == self.position:
            the_plot.add_reward(1)
            the_plot.terminate_episode()


class DummyDrape(pycolab.things.Drape):
    """Drape that does nothing."""

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del actions, board, layers, backdrop, things, the_plot


class ConstantRewardDrape(pycolab.things.Drape):
    """Drape for fields with constant rewards.

    When it's reached by the player, gives a reward.
    """

    def __init__(self, curtain, character, reward_matrix):
        super(ConstantRewardDrape, self).__init__(curtain, character)
        self._reward_matrix = reward_matrix

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del actions, board, layers, backdrop
        if self.curtain[things['P'].position]:
            the_plot.add_reward(self._reward_matrix[things['P'].position])


class OctoGrid(PyColabEnv):
    """OctoMaze on a grid.

    The number and length of corridors is configurable. Only one corridor is
    passable, the rest are blocked at random depths.

    At the beginning of each episode, the passable corridor and the depths of
    the other corridors are randomized.

    If soft is set to True, the agent gets random rewards from uniform
    distribution every time he steps in a dead-end corridor.
    """

    allowed_stochasticities = [
        base.Stochasticity.none,
        base.Stochasticity.episodic,
    ]

    def __init__(
        self,
        n_corridors=3,
        corridor_length=4,
        stochasticity=base.Stochasticity.episodic,
        init_seed=0,
        soft=False,
        lower_bound=-0.1,
        upper_bound=0,
    ):
        self._n_corridors = n_corridors
        self._corridor_length = corridor_length

        self._soft = soft
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        super().__init__(
            n_actions=4,
            default_reward=0,
            # Hide the corridor blocks.
            hidden_characters={'-', 'r'},
            stochasticity=stochasticity,
            init_seed=init_seed,
        )

    def _generate_board(self):
        if self.np_random is None:
            self.seed(self._init_seed)

        # Randomize the depths.
        depths = [
            self.np_random.randint(self._corridor_length)
            for _ in range(self._n_corridors)
        ]
        # Choose the passable corridor at random.
        corridor = self.np_random.randint(self._n_corridors)
        depths[corridor] = self._corridor_length

        horizontal_wall = '#' * (self._n_corridors * 2 + 1)
        half_space = ' ' * (self._n_corridors - 1)

        inner_width = self._n_corridors * 2 - 1
        first_row = ['#'] + [' '] * inner_width + ['#']
        # Randomize the initial player's position.
        first_row[self.np_random.randint(inner_width) + 1] = 'P'
        first_row = ''.join(first_row)

        def row(i):
            # Put a '-' (invisible block) where each corridor ends.
            # Only in soft version:
            #   Put a 'r' (invisible reward block) in every dead-end corridor.
            row_string = '#'
            for j in range(self._n_corridors):
                if depths[j] == i:
                    row_string += '-'
                elif j == corridor:
                    row_string += ' '
                elif self._soft:
                    row_string += 'r'
                else:
                    row_string += ' '
                row_string += '#'

            return row_string

        return [
            #  #####
            #  # P #
            horizontal_wall,
            first_row,
        ] + [
            #  #r# #
            #  #-# #
            #  # # #
            row(i) for i in range(self._corridor_length)
        ] + [
            #  # X #
            #  #####
            '#' + half_space + 'X' + half_space + '#',
            horizontal_wall,
        ]

    def make_game(self):
        if self.np_random is None:
            self.seed(self._init_seed)

        board = self._generate_board()
        matrix_shape = (len(board), len(board[0]))
        reward_matrix = self.np_random.uniform(self._lower_bound,
                                                self._upper_bound,
                                                matrix_shape)
        return pycolab.ascii_art.ascii_art_to_game(
            board,
            what_lies_beneath=' ',
            sprites={
                'P': PlayerSprite,
                'X': GoalSprite,
            },
            drapes={
                '-': DummyDrape,
                'r': pycolab.ascii_art.Partial(ConstantRewardDrape,
                                               reward_matrix)
            },
            update_schedule=[['r'], ['P'], ['X'], ['-']],
        )


class MudWalk(PyColabEnv):
    """ Grid world with negative rewards.

    The game is an empty room. The player starts in the upper left corner and
    the goal is in the bottom right corner. When stepping on a field, player
    receives a reward (negative by default) that is randomly chosen at the
    beginning of the episode.
    """

    allowed_stochasticities = [
        base.Stochasticity.none,
        base.Stochasticity.episodic,
    ]

    def __init__(
        self,
        board_shape=(4, 4),
        stochasticity=base.Stochasticity.episodic,
        lower_bound=-0.1,
        upper_bound=0,
        init_seed=0,
    ):
        assert board_shape[0] > 1 and board_shape[1] > 1
        self._board_shape = board_shape
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        super().__init__(
            n_actions=4,
            default_reward=0,
            hidden_characters={'r'},
            stochasticity=stochasticity,
            init_seed=init_seed,
        )

    def _generate_board(self):
        height, width = self._board_shape
        horizontal_border = '#' * (width + 2)
        first_row = '#' + 'P' + 'r' * (width - 1) + '#'
        middle_row = '#' + 'r' * width + '#'
        last_row = '#' + 'r' * (width - 1) + 'X' + '#'

        return [
            #  ######
            #  #Prrr#
            horizontal_border,
            first_row,
        ] + [
            #  #rrrr#
            middle_row for _ in range(height - 2)
        ] + [
            #  #rrrX#
            #  ######
            last_row,
            horizontal_border,
        ]

    def make_game(self):
        if self.np_random is None:
            self.seed(self._init_seed)

        board = self._generate_board()
        matrix_shape = (len(board), len(board[0]))
        reward_matrix = self.np_random.uniform(self._lower_bound,
                                                self._upper_bound,
                                                matrix_shape)

        return pycolab.ascii_art.ascii_art_to_game(
            board,
            what_lies_beneath=' ',
            sprites={
                'P': PlayerSprite,
                'X': GoalSprite,
            },
            drapes={
                'r': pycolab.ascii_art.Partial(ConstantRewardDrape,
                                               reward_matrix)
            },
            update_schedule=[['r'], ['P'], ['X']],
        )


class TwoCorridors(PyColabEnv):
    """ Grid world with two corridors.

    There are two corridors with different lengths. Both lead to the Goal of the
    game. If shorter_corridor_open is set to True the the shorter corridor is
    open, otherwise it is closed. Player receives -1 reward for each step.
    """

    allowed_stochasticities = [
        base.Stochasticity.none,
    ]

    def __init__(
        self,
        corridor_length=5,
        shorter_corridor_open=True,
        stochasticity=base.Stochasticity.none,

        init_seed=0,
    ):
        self._corridor_length = corridor_length
        self._shorter_corridor_open = shorter_corridor_open

        super().__init__(
            n_actions=4,
            default_reward=0,
            solved_at_reward=None,
            stochasticity=stochasticity,
            init_seed=init_seed,
        )

    def _generate_board(self):
        width = 7
        horizontal_border = '#' * width
        first_row = '#' + 'P' + ' ' * (width - 3) + '#'
        middle_row = '#' + ' ' + '#' * (width - 4) + ' ' + '#'
        closed_row = '#' * (width - 2) + ' ' + '#'
        last_row = '#' + 'X' + ' ' * (width - 3) + '#'

        return [
            #  #######
            #  #P    #
            horizontal_border,
            first_row,
        ] + [
            #  # ### #
            middle_row for _ in range(self._corridor_length - 1)
        ] + [
            #  # ### #  or  ##### #
            middle_row if self._shorter_corridor_open else closed_row
        ] + [
            #  #X    #
            #  #######
            last_row,
            horizontal_border,
        ]

    def make_game(self):
        if self.np_random is None:
            self.seed(self._init_seed)

        board = self._generate_board()
        return pycolab.ascii_art.ascii_art_to_game(
            board,
            what_lies_beneath=' ',
            sprites={
                'P': pycolab.ascii_art.Partial(PlayerSprite, -1),
                'X': GoalSprite,
            },
            update_schedule=[['P'], ['X']],
        )
