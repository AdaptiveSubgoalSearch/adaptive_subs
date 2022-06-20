"""Monte Carlo simulations agents."""

import math

import gin
import gym
import numpy as np

from alpacka import batch_steppers
from alpacka import data
from alpacka.agents import base
from alpacka.agents import core
from alpacka.utils import metric as metric_utils
from alpacka.utils import transformations


@gin.configurable
def truncated_return(episodes, discount=1.):
    """Returns sum of rewards up to the truncation of the episode."""
    return np.array([
        transformations.discount_cumsum(
            episode.transition_batch.reward, discount
        )[0]
        for episode in episodes
    ])

    # To indicate it's a coroutine:
    yield  # pylint: disable=unreachable


@gin.configurable
def bootstrap_return_with_value(episodes, discount=1.):
    """Bootstraps a state value at the end of the episode if truncated."""
    last_values, _ = yield np.array([
        episode.transition_batch.next_observation[-1]
        for episode in episodes
    ])
    last_values = np.squeeze(last_values, axis=1)

    # Apply discount to values from network.
    last_values = np.array([
        last_value * discount ** len(episode.transition_batch.reward)
        for last_value, episode in zip(last_values, episodes)
    ], dtype=np.float32)

    returns_ = [
        transformations.discount_cumsum(
            episode.transition_batch.reward, discount
        )[0]
        for episode in episodes
    ]

    returns_ = [
        return_ + last_value if episode.truncated else return_
        for episode, return_, last_value in
        zip(episodes, returns_, last_values)
    ]
    return returns_


class MCSimulationAgent(base.OnlineAgent):
    """Base for Monte Carlo agents."""

    def __init__(
        self,
        n_rollouts=1000,
        rollout_time_limit=None,
        estimate_fn=truncated_return,
        batch_stepper_class=batch_steppers.LocalBatchStepper,
        agent_class=core.RandomAgent,
        n_envs=10,
        discount=1.,
        **kwargs
    ):
        """Initializes MCSimulationAgent.

        Args:
            n_rollouts (int): Do at least this number of MC rollouts per act().
            rollout_time_limit (int): Maximum number of timesteps for rollouts.
            estimate_fn (callable): Simulated episode return estimator.
                Signature: Episode -> float: return. It must be a coroutine.
            batch_stepper_class (type): BatchStepper class.
            agent_class (type): Rollout agent class.
            n_envs (int): Number of parallel environments to run.
            discount (float): Future reward discount factor (also known as
                gamma)
            kwargs: OnlineAgent init keyword arguments.
        """
        super().__init__(**kwargs)
        self._n_rollouts = n_rollouts
        self._rollout_time_limit = rollout_time_limit
        self._estimate_fn = estimate_fn
        self._batch_stepper_class = batch_stepper_class
        self._agent_class = agent_class
        self._n_envs = n_envs
        self._discount = discount
        self._model = None
        self._batch_stepper = None
        self._network_fn = None
        self._params = None

        self._sim_agent = agent_class()

    def reset(self, env, observation):
        """Reinitializes the agent for a new environment."""
        assert isinstance(env.action_space, gym.spaces.Discrete), (
            'MCSimulationAgent only works with Discrete action spaces.'
        )
        yield from super().reset(env, observation)

        self._model = env
        self._network_fn, self._params = yield data.NetworkRequest()

    def act(self, observation):
        """Runs simulations and chooses the best action."""
        assert self._model is not None, (
            'Reset MCSimulationAgent first.'
        )

        # Run a prior agent.
        yield from self._sim_agent.reset(self._model, observation)
        _, prior_info = yield from self._sim_agent.act(observation)

        # Lazy initialize batch stepper
        if self._batch_stepper is None:
            self._batch_stepper = self._batch_stepper_class(
                env_class=type(self._model),
                agent_class=self._agent_class,
                network_fn=self._network_fn,
                n_envs=self._n_envs,
                output_dir=None,
            )

        action, agent_info = yield from self._act(observation, prior_info)

        # Calculate simulation policy entropy, average value and logits.
        if 'entropy' in prior_info:
            agent_info['simulation_entropy'] = prior_info['entropy']
        if 'value' in prior_info:
            agent_info['network_value'] = prior_info['value']
        if 'logits' in prior_info:
            agent_info['network_logits'] = prior_info['logits']

        return action, agent_info

    def network_signature(self, observation_space, action_space):
        return self._sim_agent.network_signature(
            observation_space, action_space)

    def postprocess_transitions(self, transitions):
        rewards = [transition.reward for transition in transitions]
        discounted_returns = transformations.discount_cumsum(
            rewards, self._discount
        )

        for transition, discounted_return in zip(
            transitions, discounted_returns
        ):
            transition.agent_info['discounted_return'] = discounted_return

        return transitions

    def _act(self, observation, prior_info):
        raise NotImplementedError()
        # To indicate it's a coroutine:
        yield  # pylint: disable=unreachable

    def _rollout_simulation_agent(self, state=None):
        init_state = state if state is not None else self._model.clone_state()
        episodes = self._batch_stepper.run_episode_batch(
            params=self._params,
            epoch=self._epoch,
            init_state=init_state,
            time_limit=self._rollout_time_limit,
        )
        self._run_agent_callbacks(episodes)
        return episodes

    def _run_agent_callbacks(self, episodes):
        for episode in episodes:
            for callback in self._callbacks:
                callback.on_pass_begin()
                transition_batch = episode.transition_batch
                for ix in range(len(transition_batch.action)):
                    step_agent_info = {
                        key: value[ix]
                        for key, value in transition_batch.agent_info.items()
                    }
                    callback.on_model_step(
                        agent_info=step_agent_info,
                        action=transition_batch.action[ix],
                        observation=transition_batch.next_observation[ix],
                        reward=transition_batch.reward[ix],
                        done=transition_batch.done[ix]
                    )
                callback.on_pass_end()

    @staticmethod
    def compute_metrics(episodes):
        metrics = {}
        agent_info_batch = data.nested_concatenate(
            [episode.transition_batch.agent_info for episode in episodes])

        # Don't log the action histogram.
        agent_info_batch.pop('action_histogram', None)

        if 'value' in agent_info_batch:
            metrics.update(metric_utils.compute_scalar_statistics(
                agent_info_batch.pop('value'),
                prefix='simulation_value',
                with_min_and_max=True
            ))

        if 'qualities' in agent_info_batch:
            metrics.update(metric_utils.compute_scalar_statistics(
                agent_info_batch.pop('qualities'),
                prefix='simulation_qualities',
                with_min_and_max=True
            ))

        # Log the rest of metrics from agent info.
        for prefix, x in agent_info_batch.items():
            metrics.update(metric_utils.compute_scalar_statistics(
                x, prefix=prefix, with_min_and_max=True
            ))

        return metrics

    def close(self):
        """Closes the batch stepper."""
        self._batch_stepper.close()


# Basic returns aggregators.
gin.external_configurable(np.max, module='np')
gin.external_configurable(np.mean, module='np')


class ShootingAgent(MCSimulationAgent):
    """Monte Carlo prediction agents.

    Uses first-visit Monte Carlo prediction for estimating action qualities
    in the current state.
    """

    def __init__(
        self,
        aggregate_fn=np.mean,
        **kwargs
    ):
        """Initializes ShootingAgent.

        Args:
            aggregate_fn (callable): Aggregates simulated episodes returns.
                Signature: list: returns -> float: action score.
            kwargs: MCSimulationAgent init keyword arguments.
        """
        super().__init__(**kwargs)
        self._aggregate_fn = aggregate_fn

    def _act(self, observation, prior_info):
        del observation
        del prior_info

        episodes = []
        for _ in range(math.ceil(self._n_rollouts / self._n_envs)):
            episodes.extend(self._rollout_simulation_agent())

        # Compute episode returns and put them in a map.
        returns_ = yield from self._estimate_fn(episodes, self._discount)
        act_to_rets_map = {key: [] for key in range(self._action_space.n)}
        for episode, return_ in zip(episodes, returns_):
            act_to_rets_map[episode.transition_batch.action[0]].append(return_)

        # Aggregate episodes into action scores.
        action_scores = np.empty(self._action_space.n)
        for action, returns in act_to_rets_map.items():
            action_scores[action] = (self._aggregate_fn(returns)
                                     if returns else np.nan)

        # Calculate the estimate of a state value.
        value = sum(returns_) / len(returns_)

        # Choose greedy action (ignore NaN scores).
        action = np.nanargmax(action_scores)
        onehot_action = np.zeros_like(action_scores)
        onehot_action[action] = 1

        # Pack statistics into agent info.
        agent_info = {
            'action_histogram': onehot_action,
            'value': value,
            'qualities': action_scores,
        }

        return action, agent_info


@gin.configurable
def ucb_bonus(action_counts, **kwargs):
    """Calculates Upper Confidence Bound bonus for all actions.

    **kwargs absorbs all possible additional information about actions.
    """
    del kwargs

    total_count = np.sum(action_counts)
    return np.sqrt(total_count) / (1 + action_counts)


@gin.configurable
def pucb_bonus(action_counts, *, prior_probs, **kwargs):
    """Calculates UCB bonus weighted by prior probabilities for all actions.

    **kwargs absorbs all possible additional information about actions.
    """
    del kwargs

    total_count = np.sum(action_counts)
    return prior_probs * np.sqrt(total_count) / (1 + action_counts)


class BanditAgent(MCSimulationAgent):
    """Flat-bandit agent.

    Uses quality + bonus formula for estimating action utilities in each bandit
    iteration and, at the end, calculates action probabilities from action
    counts in the current state.
    """

    def __init__(
        self,
        temperature=0.0,
        prior_noise=None,
        noise_weight=0.25,
        bonus_fn=pucb_bonus,
        exploration_weight=2.5,
        **kwargs
    ):
        """Initializes BanditAgent.

        Args:
            temperature (float): Proportional dist. temperature parameter.
            prior_noise (float): Dirichlet distribution parameter alpha.
                Controls how strong is noise added to a prior policy. If None,
                then disabled.
            noise_weight (float): Proportion in which prior noise is added.
            bonus_fn (callable): Calculates exploration bonus for all actions.
            exploration_weight (float): The parameter exploration_weight >= 0
                controls the trade-off between choosing lucrative nodes (low
                weight) and exploring nodes with low visit counts (high weight).
                Rule of thumb: 7 / avg. number of legal moves.
            kwargs: MCSimulationAgent init keyword arguments.
        """
        super().__init__(**kwargs)
        self._temperature = temperature
        self._prior_noise = prior_noise
        self._noise_weight = noise_weight
        self._bonus_fn = bonus_fn
        self._exploration_weight = exploration_weight

    def _evaluate(self, state):
        """Evaluates state via MC rollouts.

        Args:
            state (object): An environment state object.

        Yields:
            Network prediction requests.

        Returns:
            float: an estimated state value.
            int: number of steps done in the simulator.
        """
        episodes = self._rollout_simulation_agent(state)
        returns_ = yield from self._estimate_fn(episodes, self._discount)
        return np.mean(returns_), sum(map(
            lambda e: len(e.transition_batch.action), episodes))

    def _act(self, observation, prior_info):
        # Save the root state
        root_state = self._model.clone_state()

        # (Optionally) apply noise to prior and calculate info statistics
        agent_info = {}
        prior_probs = prior_info.get(
            'prob',
            np.ones(self._model.action_space.n) / self._model.action_space.n,
        )
        if self._prior_noise is not None:
            prior_noise = np.random.dirichlet(
                [self._prior_noise, ] * len(prior_probs))
            # Add dirichlet noise according to AlphaZero paper.
            prior_probs = ((1 - self._noise_weight) * prior_probs +
                           self._noise_weight * prior_noise)

            agent_info.update({
                'noise_entropy': -np.nansum(prior_noise * np.log(prior_noise)),
                'sim_noise_cross_entropy': -np.nansum(
                    prior_info['prob'] * np.log(prior_noise)),
            })
        agent_info['prior_entropy'] = -np.nansum(
            prior_probs * np.log(prior_probs))

        # Run flat-UCB for n_rollouts iterations.
        action_counts = np.zeros_like(prior_probs)
        action_qualities = np.zeros_like(prior_probs)
        n_simulated_timesteps = 0
        n_bandit_iterations = 0
        for _ in range(self._n_rollouts):
            # 1. Select bandit.
            action_utilities = (action_qualities +
                                self._exploration_weight * self._bonus_fn(
                                    action_counts, prior_probs=prior_probs))
            action = np.argmax(action_utilities)

            # 2. Play bandit.
            _, reward, done = self._model.step(action)
            if done:
                value, n_steps = 0., 0
            else:
                value, n_steps = yield from self._evaluate(
                    self._model.clone_state())
            n_simulated_timesteps += (n_steps + 1)

            # 3. Update bandit.
            return_ = reward + self._discount * value
            action_counts[action] += 1
            action_qualities[action] += (
                (return_ - action_qualities[action]) / action_counts[action])

            # Restore model and book keeping.
            self._model.restore_state(root_state)
            n_bandit_iterations += 1

        # Calculate the estimate of a state value and transform qualities.
        value = np.sum(action_counts * action_qualities) / np.sum(action_counts)
        qualities = action_qualities[:]
        qualities[action_counts == 0] = np.nan

        # Calculate action prob. and entropy.
        if self._temperature > 0.01:
            action_counts_temp = action_counts ** (1 / self._temperature)
            action_probs = action_counts_temp / np.sum(action_counts_temp)
            agent_info['search_entropy'] = -np.nansum(
                action_probs * np.log(action_probs))

            # Sample action.
            action = np.random.choice(len(action_probs), p=action_probs)
        else:
            agent_info['search_entropy'] = 0

            # Choose action greedily.
            action = np.argmax(action_counts)

        # Calculate the action histogram and its entropy
        action_histogram = action_counts / np.sum(action_counts)
        histogram_entropy = -np.nansum(
            action_histogram * np.log(action_histogram))

        # Pack statistics into agent info.
        agent_info.update({
            'action_counts': action_counts,
            'action_histogram': action_histogram,
            'histogram_entropy': histogram_entropy,
            'n_bandit_iterations': n_bandit_iterations,
            'n_simulated_timesteps': n_simulated_timesteps,
            'value': value,
            'qualities': qualities,
            'simulation_bonuses': self._bonus_fn(
                action_counts, prior_probs=prior_probs),
        })

        return action, agent_info
