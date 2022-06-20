"""Base classes related to environments."""

import enum

import gin
import gym


@gin.constants_from_enum
class Stochasticity(enum.Enum):
    """Stochastiticty mode."""

    # Can't assume the stochasticity mode.
    unknown = 0
    # Deterministic environment.
    none = 1
    # Stochasticity within one episode, but not between episodes, i.e. state
    # includes the random seed.
    episodic = 2
    # Stochasticity both within one episode and between episodes, i.e. state
    # doesn't include the random seed.
    universal = 3


class RestorableEnv(gym.Env):
    """Environment interface used by model-based agents.

    This class defines an additional interface over gym.Env that is assumed by
    model-based agents. It's just for documentation purposes, doesn't have to be
    subclassed by envs used as models (but it can be).
    """

    stochasticity = Stochasticity.unknown

    def clone_state(self):
        """Returns the current environment state."""
        raise NotImplementedError

    def restore_state(self, state):
        """Restores environment state, returns the observation."""
        raise NotImplementedError


class EnvRenderer:
    """Base class for environment renderers."""

    def __init__(self, env):
        """Initializes EnvRenderer."""
        del env

    def render_state(self, state_info):
        """Renders state_info to an image."""
        raise NotImplementedError

    def render_heatmap(self, heatmap, current_state_info):
        """Renders a state_info -> float mapping to an image."""
        raise NotImplementedError

    def render_action(self, action):
        """Renders action to a string."""
        raise NotImplementedError
