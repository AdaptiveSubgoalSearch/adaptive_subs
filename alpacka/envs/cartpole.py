"""CartPole env."""

import numpy as np
from gym.envs import classic_control

from alpacka.envs import base


try:
    import cv2
except ImportError:
    cv2 = None


class CartPole(classic_control.CartPoleEnv, base.RestorableEnv):
    """CartPole with state clone/restore and returning a "solved" flag."""

    stochasticity = base.Stochasticity.episodic

    def __init__(self, solved_at=500, reward_scale=1., **kwargs):
        super().__init__(**kwargs)

        self.solved_at = solved_at
        self.reward_scale = reward_scale

        self._step = None

    def reset(self):
        self._step = 0
        return super().reset()

    def step(self, action):
        (observation, reward, done, info) = super().step(action)
        info['solved'] = self._step >= self.solved_at
        self._step += 1
        return (observation, reward * self.reward_scale, done, info)

    def clone_state(self):
        return (tuple(self.state), self.steps_beyond_done, self._step)

    def restore_state(self, state):
        (state, self.steps_beyond_done, self._step) = state
        self.state = np.array(state)
        return self.state

    class Renderer(base.EnvRenderer):
        """Renderer for CartPole.

        Uses cv2 for state rendering.
        """

        def __init__(self, env):
            """Defines rendering paramters as in the original Gym CartPole."""
            super().__init__(env)

            self._screen_width = 600
            self._screen_height = 400
            self._world_width = 2 * 2.4
            self._scale = self._screen_width / self._world_width
            self._cart_y = self._screen_height - 100
            self._pole_width = 10
            self._pole_len = self._scale
            self._cart_width = 50
            self._cart_height = 30

        def render_state(self, state_info):
            if cv2 is None:
                raise ImportError('Could not import cv2! '
                                  'HINT: Install Alpacka with TraceX extras. '
                                  'See alpacka/tools/tracex/README.md')

            # Get the cart position and the pole angle.
            position, _, angle, _ = state_info
            cart_x = int(position * self._scale + self._screen_width / 2.0)

            # Prepare a blank image of screen size.
            img = np.ones((self._screen_height, self._screen_width, 3)) * 255.

            # Draw a rail.
            img = cv2.line(  # pylint: disable=no-member
                img,
                pt1=(0, self._cart_y),
                pt2=(self._screen_width, self._cart_y),
                color=(0, 0, 0)
            )

            # Draw a cart.
            img = cv2.rectangle(  # pylint: disable=no-member
                img,
                pt1=(cart_x - self._cart_width // 2,
                     self._cart_y - self._cart_height // 2),
                pt2=(cart_x + self._cart_width // 2,
                     self._cart_y + self._cart_height // 2),
                color=(0, 0, 0),
                thickness=cv2.FILLED  # pylint: disable=no-member
            )

            # Draw a pole.
            img = cv2.line(  # pylint: disable=no-member
                img,
                pt1=(cart_x, self._cart_y),
                pt2=(int(cart_x + self._pole_len * np.sin(angle)),
                     int(self._cart_y - self._pole_len * np.cos(angle))),
                color=(204, 153, 102),
                thickness=self._pole_width
            )

            # Draw an axle.
            img = cv2.circle(  # pylint: disable=no-member
                img,
                center=(cart_x, self._cart_y),
                radius=self._pole_width // 2,
                color=(127, 127, 204),
                thickness=cv2.FILLED  # pylint: disable=no-member
            )

            return img.astype(np.uint8)

        def render_action(self, action):
            return ['left', 'right'][action]
