"""Environment's models used by model-based agents."""

from alpacka import data


class EnvModel:
    """Environment's model interface for model-based algorithms.

    All derived classes should set a class-level attribute:
        is_perfect (bool): Info for the agent, if they can trust the model
            or if they should be prepared for potential mispredictions.
    """

    def __init__(self, env):
        """Creates a model.

        Args:
            env (RestorableEnv): Modeled environment.
        """
        self._action_space = env.action_space
        self._observation_space = env.observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def step(self, action):
        """Performs one step on the model.

        It do change model's state - as opposed to predict_steps() method.

        Args:
            action: Action to perform.

        Yields:
            A stream of Network inputs requested for inference.

        Returns:
            Tuple (observation, reward, done). It doesn't return info dict,
            as opposed to gym.Env.step().
        """
        raise NotImplementedError
        yield  # pylint: disable=unreachable

    def predict_steps(self, actions, include_state):
        """Predicts environment's behaviour on the given set of actions.

        It doesn't change model's state - as opposed to step() method.

        Args:
            actions (list): List of actions to simulate.
            include_state (bool): Whether to include states in the returned
                data.

        Yields:
            A stream of Network inputs requested for inference.

        Returns:
            Tuple (observations, rewards, dones) or
            (observations, rewards, dones, states) of the successors of the
            current state of the model (depending on include_state).
        """
        raise NotImplementedError
        yield  # pylint: disable=unreachable

    def catch_up(self, observation):
        """Catches up the model with the state of environment being solved.

        When the state of real env changes (e.g. when OnlineAgent calls env's
        step() method), this method synchronizes the state of EnvModel with
        the given observation (coming from the real env).

        Overriding is not needed for models, which use exactly the real env
        instance that Agent is solving.

        Models, which use independent env instance or doesn't use env at all,
        should override this method.
        """

    def correct(self, obs, action, next_obs, reward, done, agent_info):
        """Corrects potential model's mispredictions using data from real env.

        Model may digest the given transition not to repeat the same mistake
        twice. Transition should come from the real env - so it should be
        correct.

        Overriding is optional. In particular, perfect models don't need that
        at all.
        """

    def clone_state(self):
        """Returns the current model state."""
        raise NotImplementedError

    def restore_state(self, state):
        """Restores model state, returns the observation."""
        raise NotImplementedError


class PerfectModel(EnvModel):
    """Simple wrapper around RestorableEnv implementing EnvModel interface."""
    is_perfect = True

    def __init__(self, env):
        super().__init__(env)
        self._env = env

    def step(self, action):
        return self._env.step(action)[:-1]
        yield  # pylint: disable=unreachable

    def predict_steps(self, actions, include_state):
        return step_into_successors(self._env, actions, include_state)
        yield  # pylint: disable=unreachable

    def clone_state(self):
        return self._env.clone_state()

    def restore_state(self, state):
        return self._env.restore_state(state)


def step_into_successors(env, actions, include_state):
    """Explores the successors of the current state of the environment.

    Args:
        env (RestorableEnv): The environment.
        actions (list): List of actions to check.
        include_state (bool): Whether to include states in the returned data.

    Returns:
        Same as for EnvModel.predict_steps().
    """
    init_state = env.clone_state()

    def step_and_rewind(action):
        (observation, reward, done, _) = env.step(action)
        if include_state:
            state = env.clone_state()
        env.restore_state(init_state)
        info = (observation, reward, done)
        if include_state:
            info += (state,)
        return info

    (observations, rewards, dones, *maybe_states) = list(zip(*[
        step_and_rewind(action) for action in actions
    ]))
    return list(map(
        data.nested_stack, (observations, rewards, dones)
    )) + maybe_states
