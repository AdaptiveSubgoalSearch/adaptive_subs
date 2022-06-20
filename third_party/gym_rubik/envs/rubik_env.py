from enum import Enum
import time

import gin
import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

from gym_rubik.envs.cube import Actions, Cube
from gym_rubik.envs.converter import CubeConverter


class DebugLevel(Enum):
    WARNING = 0,
    INFO = 1,
    VERBOSE = 2


@gin.configurable
class RubikEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, step_limit=100, shuffles=50, obs_type='basic'):
        self.cube = Cube(3, whiteplastic=False)
        self.action_space = spaces.Discrete(len(ACTION_LOOKUP))
        self.fig = None
        self.solved_state = self.cube.get_state()
        
        self.observation_space = None
        self.obs_type = obs_type
        self.converter = None
        self.create_observation_space()

        self.scramble = []

        self.debugLevel = DebugLevel.WARNING
        self.renderViews = True
        self.renderFlat = True
        self.renderCube = False
        self.scrambleSize = shuffles
        
        self.num_steps = 0
        self.step_limit = step_limit

        self.solved_obs = self._get_state()

        self.config()

    def config(self, debug_level=DebugLevel.WARNING, render_cube=False, scramble_size=None, render_views=True,
               render_flat=True, step_limit=None):
        self.debugLevel = debug_level
        self.renderCube = render_cube
        if scramble_size is not None:
            self.scrambleSize = scramble_size
        if step_limit is not None:
            self.step_limit = step_limit

        self.renderViews = render_views
        self.renderFlat = render_flat

        if self.renderCube:
            plt.ion()
            plt.show()
    
    def create_observation_space(self):
        if self.obs_type == 'basic':
            self.observation_space = spaces.Box(low=0, high=1, shape=(6, 3, 3, 6), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0, high=1, shape=(20, 24), dtype=np.float32)
            self.converter = CubeConverter()

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        observation, reward, episode_over, info : tuple
            observation (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self._take_action(action)
        reward = -1
        self.num_steps += 1

        observation = self._get_state()
        solved = np.array_equal(self.cube.get_state(), self.solved_state)

        if solved:
            reward = 0
        
        episode_over = solved or (self.num_steps == self.step_limit)
        
        return observation, reward, episode_over, {}

    def reset(self):
        # print(self.scrambleSize)
        self.cube = Cube(3, whiteplastic=False)
        self.scramble = []
        if self.scrambleSize > 0:
            if self.debugLevel == DebugLevel.INFO:
                print("scramble " + str(self.scrambleSize) + " moves")
            self.randomize(self.scrambleSize)

        self.num_steps = 0
        return self._get_state()

    def render(self, mode='human', close=False):
        if self.renderCube:
            if self.fig:
                plt.clf()
            self.fig = self.cube.render(self.fig, views=self.renderViews, flat=self.renderFlat)
            plt.pause(0.001)

    def _take_action(self, action):
        self.cube.move_by_action(ACTION_LOOKUP[action])

    @staticmethod
    def action_name(action):
        return ACTION_LOOKUP[action].name

    def get_scramble(self):
        return self.scramble

    def valid_scramble_action(self, action, previous_actions):
        num_previous_actions = len(previous_actions)
        if num_previous_actions > 2 \
                and previous_actions[num_previous_actions - 1] == previous_actions[num_previous_actions - 2] \
                and action.name == previous_actions[num_previous_actions - 1]:
            return False
        if num_previous_actions > 1 \
                and self.cube.opposite_actions(previous_actions[num_previous_actions - 1], action):
            return False
        return True

    def randomize(self, number):
        t = 0
        while t < number:
            action = ACTION_LOOKUP[np.random.randint(len(ACTION_LOOKUP.keys()))]
            if self.valid_scramble_action(action, self.scramble):
                self.scramble.append(action.name)
                self.cube.move_by_action(action)
                t += 1

    def _get_state(self):
        raw_state = self.cube.get_state()
        if self.obs_type == 'basic':
            state = (np.arange(6) == raw_state[..., np.newaxis]).astype(int)
        else:
            state = self.converter.convert_basic_to_reduced(raw_state)
        return state

    def load_state(self, desired_state):
        assert desired_state.shape == (6, 3, 3)
        self.cube.stickers = desired_state


ACTION_LOOKUP = {
    0: Actions.U,
    1: Actions.U_1,
    2: Actions.D,
    3: Actions.D_1,
    4: Actions.F,
    5: Actions.F_1,
    6: Actions.B,
    7: Actions.B_1,
    8: Actions.R,
    9: Actions.R_1,
    10: Actions.L,
    11: Actions.L_1
}


@gin.configurable
class GoalRubikEnv(RubikEnv):
    def __init__(self, step_limit=100, shuffles=50, obs_type='basic'):
        super(GoalRubikEnv, self).__init__(step_limit, shuffles, obs_type)
        self.goal_obs = self._get_state()

    def create_observation_space(self):
        if self.obs_type == 'basic':
            self.observation_space = spaces.Box(low=0, high=1, shape=(6, 3, 3, 12), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0, high=1, shape=(20, 48), dtype=np.float32)
            self.converter = CubeConverter()

    def step(self, action):
        obs, reward, done, info = super(GoalRubikEnv, self).step(action)

        obs = self._get_goal_observation(obs)
        reward = self._calculate_reward(obs['observation'], obs['achieved_goal'], obs['desired_goal'])

        return obs, reward, done, info

    def reset(self):
        obs = super(GoalRubikEnv, self).reset()
        # print(obs.flatten())
        # print(self._get_goal_observation(obs))
        return self._get_goal_observation(obs)

    def _get_goal_observation(self, obs):
        return self._convert_observation(obs, obs, self.goal_obs)

    def _convert_observation(self, obs, state, goal):
        # print(obs.shape, state.shape, goal.shape)
        return {'observation': obs, 'achieved_goal': state, 'desired_goal': goal}

    def _calculate_reward(self, obs, state, goal):
        return 0 if np.array_equal(state, goal) else -1

    def set_goal(self, goal_obs):
        self.goal_obs = goal_obs
