"""Agent base classes."""

from alpacka import data
# from alpacka import envs
from alpacka import metric_logging
from alpacka import utils
from alpacka.utils import space


class Agent:
    """Agent base class.

    Agents can use neural networks internally. Network prediction is run outside
    of the Agent, so it can be batched across multiple Agents for efficiency.
    This is done using a coroutine API, explained in solve().
    """

    def __init__(self, parameter_schedules=None):
        """Initializes Agent.

        Args:
            parameter_schedules (dict): Dictionary from recursive attribute name
                e.g. 'distribution.temperature' to a function (function object)
                with a signature: int: epoch -> float: value.
        """
        self._parameter_schedules = parameter_schedules or {}

    def solve(self, init_state, time_limit, epoch=None):  # pylint: disable=redundant-returns-doc,redundant-yields-doc
        """Solves a given environment.

        Coroutine, suspends execution for every neural network prediction
        request. This enables a very convenient interface for requesting
        predictions by the Agent:

            def solve(self, env, epoch=None, init_state=None, time_limit=None):
                # Planning...
                predictions = yield inputs
                # Planning...
                predictions = yield inputs
                # Planning...
                return episode

        Example usage:

            coroutine = agent.solve(env)
            try:
                # get inputs from agent.solve
                prediction_request = next(coroutine)
                network_output = process_request(prediction_request)
                # send preditions to agent.solve
                prediction_request = coroutine.send(network_output)
                # Possibly more prediction requests...
            except StopIteration as e:
                episode = e.value

        Agents overriding solve() should call `yield from super().solve()` at
        the beginning, 1. to update the hyperparameter schedules, and 2. so
        Python knows to treat it as a coroutine even if it doesn't have any
        yield statement.

        We use coroutines for different purposes than asyncio, the most
        mainstream use-case in Python. For a quick summary of coroutines and
        their use in asyncio, please refer to
        http://masnun.com/2015/11/13/python-generators-coroutines-native-coroutines-and-async-await.html.

        Args:
            env (gym.Env): Environment to solve.
            epoch (int): Current training epoch or None if no training.
            init_state (object): Reset the environment to this state.
                If None, then do normal gym.Env.reset().
            time_limit (int or None): Maximum number of steps to make on the
                solved environment. None means no time limit.

        Yields:
            A stream of Network inputs requested for inference.

        Returns:
            (Agent/Trainer-specific) Episode object summarizing the collected
            data for training the TrainableNetwork.
        """
        del init_state
        del time_limit
        for attr_name, schedule in self._parameter_schedules.items():
            param_value = schedule(epoch)
            utils.recursive_setattr(self, attr_name, param_value)
            metric_logging.log_scalar(
                'agent_param/' + attr_name, epoch, param_value
            )

        # To indicate it's a coroutine:
        return
        # yield  # pylint: disable=unreachable

    def network_signature(self, observation_space, action_space):  # pylint: disable=redundant-returns-doc,useless-return
        """Defines the signature of networks used by this Agent.

        Overriding is optional.

        Args:
            observation_space (gym.Space): Environment observation space.
            action_space (gym.Space): Environment action space.

        Returns:
            None: if the agent doesn't use a network.
            NetworkSignature: if the agent uses a single network.
            dict(type -> NetworkSignature): if the agent uses multiple networks.
                The dict maps request types to network signatures. Intended to
                use with UnionNetwork.
        """
        del observation_space
        del action_space
        return None

    def close(self):
        """Cleans up the resources, e.g. shuts down the parallel workers."""


class OnlineAgent(Agent):
    """Base class for online agents, i.e. planning on a per-action basis.

    Provides a default implementation of Agent.solve(), returning a Transition
    object with the collected batch of transitions.
    """

    def __init__(self, callback_classes=(), **kwargs):
        super().__init__(**kwargs)

        self._action_space = None
        self._epoch = None
        self._callbacks = [
            callback_class(self) for callback_class in callback_classes
        ]

    def reset(self, observation):  # pylint: disable=missing-param-doc
        """Resets the agent state.

        Called for every new environment to be solved. Overriding is optional.

        Args:
            env (gym.Env): Environment to solve.
            observation (Env-dependent): Initial observation returned by
                env.reset().

        Yields:
            A stream of Network inputs requested for inference.
        """
        del observation

        # To indicate it's a coroutine:
        return
        # yield  # pylint: disable=unreachable

    def act(self, observation):
        """Determines the next action to be performed.

        Coroutine, suspends execution similarly to Agent.solve().

        In model-based agents, the original environment state MUST be restored
        in the end of act(). This is not checked at runtime, since it would be
        a big overhead for heavier environments.

        Args:
            observation (Env-dependent): Observation from the environment.

        Yields:
            A stream of Network inputs requested for inference.

        Returns:
            Pair (action, agent_info), where action is the action to make in the
            environment and agent_info is a dict of additional info to be put as
            Transition.agent_info.
        """
        raise NotImplementedError

    def postprocess_transitions(self, transitions):
        """Postprocesses Transitions before passing them to Trainer.

        Can be overridden in subclasses to customize data collection.

        Called after the episode has finished, so can incorporate any
        information known only in the hindsight to the transitions.

        Args:
            transitions (List of Transition): Transitions to postprocess.

        Returns:
            List of postprocessed Transitions.
        """
        return transitions

    @staticmethod
    def compute_metrics(episodes):
        """Computes scalar metrics based on collected Episodes.

        Can be overridden in subclasses to customize logging in Runner.

        Called after the episodes has finished, so can incorporate any
        information known only in the hindsight to the episodes.

        Args:
            episodes (List of Episode): Episodes to compute metrics base on.

        Returns:
            Dict with metrics names as keys and metrics values as... values.
        """
        del episodes
        return {}

    def solve(self, init_state, time_limit, epoch=None):
        """Solves a given environment using OnlineAgent.act().

        Args:
            env (gym.Env): Environment to solve.
            epoch (int): Current training epoch or None if no training.
            init_state (object): Reset the environment to this state.
                If None, then do normal gym.Env.reset().
            time_limit (int or None): Maximum number of steps to make on the
                solved environment. None means no time limit.

        Yields:
            Network-dependent: A stream of Network inputs requested for
            inference.

        Returns:
            data.Episode: Episode object containing a batch of collected
            transitions and the return for the episode.
        """
        super().solve(init_state, time_limit, epoch=epoch)

        self._epoch = epoch

        observation = init_state
        self.reset(observation)

        for callback in self._callbacks:
            callback.on_episode_begin(None, observation, epoch)

        transitions = []
        done = False
        steps_completed = 0
        while not done and steps_completed < time_limit:
            step_info = self.act(observation)
            if step_info is None:
                break
            steps_completed += 1
            (action, new_subgoal, multi_step_info, agent_info) = step_info
            next_observation = new_subgoal
            reward = multi_step_info.reward
            done = multi_step_info.done

            for callback in self._callbacks:
                callback.on_real_step(
                    agent_info, action, next_observation, reward, done
                )

            transitions.append(data.Transition(
                observation=observation,
                action_list=multi_step_info.action_list,
                reward=reward,
                done=done,
                next_observation=next_observation,
                agent_info=agent_info,
            ))
            observation = next_observation

        for callback in self._callbacks:
            callback.on_episode_end()

        transitions = self.postprocess_transitions(transitions)

        return_ = sum(transition.reward for transition in transitions)
        return data.Episode(
            transitions=transitions,
            return_=return_,
            solved=done,
        )


class AgentCallback:
    """Base class for agent callbacks."""
    def __init__(self, agent):
        self._agent = agent

    # Events for all OnlineAgents.

    def on_episode_begin(self, env, observation, epoch):
        """Called in the beginning of a new episode."""

    def on_episode_end(self):
        """Called in the end of an episode."""

    def on_real_step(self, agent_info, action, observation, reward, done):
        """Called after every step in the real environment."""

    # Events only for model-based agents.

    def on_pass_begin(self):
        """Called in the beginning of every planning pass."""

    def on_pass_end(self):
        """Called in the end of every planning pass."""

    def on_model_step(self, agent_info, action, observation, reward, done):
        """Called after every step in the model."""
