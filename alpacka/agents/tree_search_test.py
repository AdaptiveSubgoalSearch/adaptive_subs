"""Tests for TreeSearchAgent."""

import numpy as np
import pytest

from alpacka import testing
from alpacka.agents import mcts_test
from alpacka.agents import mets_test
from alpacka.agents import models


all_tree_search_agents = mcts_test.all_mctses + mets_test.all_metses


def create_path_transitions(n_states, reward_fn=None, done_fn=None):
    """Creates transitions for env which looks like a simple path.

    The env has only a single action and looks as follows:
    0 -> 1 -> 2 -> ... -> (n_states - 2) -> (n_states - 1)
    (and there is a loop transition for the state (n_states - 1))

    By default for all transitions:
        reward == 0,
        done == False,
    except the transition (n_states - 2) -> (n_states - 1), which has:
        reward == 1,
        done == True.
    """

    if not reward_fn:
        reward_fn = lambda state: 1 if state == n_states - 2 else 0
    if not done_fn:
        done_fn = lambda state: state == n_states - 2
    return {
        state: {
            0: (min(state + 1, n_states - 1), reward_fn(state), done_fn(state))
        }
        for state in range(n_states)
    }


def create_env(transitions):
    env = testing.TabularEnv(
        init_state=0, n_actions=1, transitions=transitions
    )
    env.reset()
    return env


class ImperfectTestModel(models.PerfectModel):
    is_perfect = False

    def catch_up(self, observation):
        # We assume that state is equivalent to observation.
        self._env.restore_state(observation)


def create_model_class(transitions):
    class ModelWithHardcodedEnv(ImperfectTestModel):
        def __init__(self, env):
            del env
            super().__init__(create_env(transitions))
    return ModelWithHardcodedEnv


@pytest.mark.parametrize('agent_fn', all_tree_search_agents)
def test_tree_deletion_on_observation_misprediction(agent_fn):
    env_transitions = create_path_transitions(n_states=8)
    model_transitions = create_path_transitions(n_states=8)
    model_transitions[1][0] = (3, 0, 0)  # Wrong state.

    env = create_env(env_transitions)
    model_class = create_model_class(model_transitions)

    agent = agent_fn(n_passes=2, model_class=model_class)
    episode = testing.run_with_dummy_network_prediction(
        agent.solve(env, epoch=0, time_limit=10),
        network_signature=agent.network_signature(
            env.observation_space, env.action_space
        )
    )

    # We use n_passes=2.
    #
    # If we used perfect model, depth of the tree would increase by 1 at every
    # act() call (until the tree encounters the terminal state) - every single
    # pass increases the depth by one and taking the step on real env decreases
    # the depth by one.
    # So depths would look as follows: [2, 3, 4, 4, 3, 2, 1]
    #
    # Our model has wrong transition 1 -> 3. Because of the misprediction of
    # the state, the tree is reset after 2nd step, so tree_depths[1] == 2
    # instead of 4. Consequently, some further depths are smaller as well.
    tree_depths = episode.transition_batch.agent_info['depth_max']  # pylint: disable=no-member
    np.testing.assert_equal(tree_depths, [2, 3, 2, 3, 3, 2, 1])


@pytest.mark.parametrize('agent_fn', all_tree_search_agents)
def test_false_positive_done(agent_fn):
    env_transitions = create_path_transitions(n_states=6)
    model_transitions = create_path_transitions(n_states=6)
    model_transitions[2][0] = (3, 0, True)  # False positive done.

    env = create_env(env_transitions)
    model_class = create_model_class(model_transitions)

    agent = agent_fn(n_passes=6, model_class=model_class)
    episode = testing.run_with_dummy_network_prediction(
        agent.solve(env, epoch=0, time_limit=8),
        network_signature=agent.network_signature(
            env.observation_space, env.action_space
        )
    )

    # Because of model's false positive for done, the agent thinks that state 3
    # is terminal - so they don't do any search beyond the state 3
    # (therefore tree_depths[0] == 3 instead of 5).
    #
    # When the agent steps into the state 3, they realize that 3 is not
    # terminal, so they expand the tree again (therefore tree_depths[3] == 2).
    tree_depths = episode.transition_batch.agent_info['depth_max']  # pylint: disable=no-member
    np.testing.assert_equal(tree_depths, [3, 2, 1, 2, 1])


@pytest.mark.parametrize('agent_fn', all_tree_search_agents)
def test_termination_on_false_negative_done(agent_fn):
    env_transitions = create_path_transitions(
        n_states=6, done_fn=lambda state: state == 2
    )
    model_transitions = create_path_transitions(
        n_states=6, done_fn=lambda state: state == 4
    )

    env = create_env(env_transitions)
    model_class = create_model_class(model_transitions)

    expected_n_steps = 3
    time_limit = 5

    agent = agent_fn(n_passes=6, model_class=model_class)
    episode = testing.run_with_dummy_network_prediction(
        agent.solve(env, epoch=0, time_limit=time_limit),
        network_signature=agent.network_signature(
            env.observation_space, env.action_space
        )
    )

    # The model thinks that the terminal state is 5, whereas the real terminal
    # state is 3. We assert that the agent terminated after 3 steps - when
    # reaching the real terminal state.
    actual_n_steps = len(episode.transition_batch.action)  # pylint: disable=no-member
    assert actual_n_steps == expected_n_steps


@pytest.mark.parametrize('agent_fn', all_tree_search_agents)
def test_reward_mistake(agent_fn):
    env_transitions = create_path_transitions(
        n_states=6, reward_fn=lambda state: 1 if state == 4 else 0
    )
    model_transitions = create_path_transitions(
        n_states=6, reward_fn=lambda state: 1 if state in [2, 4] else 0
    )

    env = create_env(env_transitions)
    model_class = create_model_class(model_transitions)

    # Smoke test. We check that no exception is thrown.
    agent = agent_fn(n_passes=3, model_class=model_class)
    testing.run_with_dummy_network_prediction(
        agent.solve(env, epoch=0, time_limit=4),
        network_signature=agent.network_signature(
            env.observation_space, env.action_space
        )
    )


@pytest.mark.parametrize('agent_fn', all_tree_search_agents)
def test_resilience_against_fake_dead_ends(agent_fn):
    n_states = 6
    env_transitions = create_path_transitions(n_states=n_states)
    model_transitions = {
        state: {0: (state, int(state == 4), state == 4)}
        for state in range(n_states)
    }

    env = create_env(env_transitions)
    model_class = create_model_class(model_transitions)

    # Smoke test. We check that no exception is thrown - in particular
    # DeadEnd exception.
    agent = agent_fn(n_passes=3, model_class=model_class)
    testing.run_with_dummy_network_prediction(
        agent.solve(env, epoch=0, time_limit=4),
        network_signature=agent.network_signature(
            env.observation_space, env.action_space
        )
    )
