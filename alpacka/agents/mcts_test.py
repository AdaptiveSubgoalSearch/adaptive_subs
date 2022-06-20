"""Tests for alpacka.agents.mcts."""

import functools

import numpy as np
import pytest

from alpacka import agents
from alpacka import envs
from alpacka import testing
from alpacka.agents import base
from alpacka.agents import models


all_mctses = [
    # DeterministicMCTSAgent
    functools.partial(
        agents.DeterministicMCTSAgent, graph_mode=False, avoid_loops=False
    ),
    functools.partial(
        agents.DeterministicMCTSAgent, graph_mode=True, avoid_loops=False
    ),
    functools.partial(
        agents.DeterministicMCTSAgent, graph_mode=True, avoid_loops=True
    ),
    # StochasticMCTSAgent
    functools.partial(
        agents.StochasticMCTSAgent, new_leaf_rater_class=functools.partial(
            agents.stochastic_mcts.RolloutNewLeafRater,
            rollout_time_limit=2,
        ),
    ),
    functools.partial(
        agents.StochasticMCTSAgent,
        new_leaf_rater_class=agents.stochastic_mcts.ValueNetworkNewLeafRater,
    ),
    functools.partial(
        agents.StochasticMCTSAgent,
        new_leaf_rater_class=functools.partial(
            agents.stochastic_mcts.ValueNetworkNewLeafRater,
            boltzmann_temperature=0.1,
        ),
    ),
    functools.partial(
        agents.StochasticMCTSAgent, new_leaf_rater_class=functools.partial(
            agents.stochastic_mcts.QualityNetworkNewLeafRater,
            use_policy=False,
        ),
    ),
    functools.partial(
        agents.StochasticMCTSAgent, new_leaf_rater_class=functools.partial(
            agents.stochastic_mcts.QualityNetworkNewLeafRater,
            use_policy=True,
        ),
    ),
]


@pytest.mark.parametrize('agent_fn', all_mctses)
def test_integration_with_cartpole(agent_fn):
    env = envs.CartPole()
    agent = agent_fn(n_passes=2)
    network_sig = agent.network_signature(
        env.observation_space, env.action_space
    )
    episode = testing.run_with_dummy_network_prediction(
        agent.solve(env), network_sig
    )
    assert episode.transition_batch.observation.shape[0]  # pylint: disable=no-member


@pytest.mark.parametrize('agent_fn', all_mctses)
def test_act_doesnt_change_env_state(agent_fn):
    env = envs.CartPole()
    agent = agent_fn(n_passes=2)
    observation = env.reset()
    testing.run_without_suspensions(agent.reset(env, observation))

    state_before = env.clone_state()
    network_sig = agent.network_signature(
        env.observation_space, env.action_space
    )
    testing.run_with_dummy_network_prediction(
        agent.act(observation), network_sig)
    state_after = env.clone_state()
    np.testing.assert_equal(state_before, state_after)


class _CartPoleIndependentModel(models.PerfectModel):
    def __init__(self, env):
        init_state = env.clone_state()
        del env

        # We use CartPole instance independent from the given env.
        env = envs.CartPole()
        env.restore_state(init_state)
        super().__init__(env)

        self._n_catch_ups = 0

    def catch_up(self, observation):
        # We assume that CartPole.step() won't be called after reaching done.
        steps_beyond_done = None
        steps = self._n_catch_ups
        self._env.restore_state((tuple(observation), steps_beyond_done, steps))

        self._n_catch_ups += 1


class _AssertModelSync(base.AgentCallback):
    def __init__(self, agent, env):
        super().__init__(agent)
        self._agent = agent
        self._env = env

    def on_pass_begin(self):
        np.testing.assert_equal(
            self._env.clone_state(),
            self._agent.model.clone_state()
        )


@pytest.mark.parametrize('agent_fn', all_mctses)
def test_synchronize_independent_model_with_real_env(agent_fn):
    env = envs.CartPole()
    callback_fn = functools.partial(_AssertModelSync, env=env)
    agent = agent_fn(
        n_passes=2,
        model_class=_CartPoleIndependentModel,
        callback_classes=[callback_fn]
    )

    network_sig = agent.network_signature(
        env.observation_space, env.action_space
    )
    testing.run_with_dummy_network_prediction(
        agent.solve(env, time_limit=2), network_sig
    )
