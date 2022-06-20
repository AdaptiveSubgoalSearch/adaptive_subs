"""Tests for alpacka.agents.stochastic_mcts."""

import functools
from unittest import mock

import pytest

from alpacka import agents
from alpacka import testing


class TabularNewLeafRater(agents.stochastic_mcts.NewLeafRater):
    """Rates new leaves based on hardcoded values."""

    def __init__(self, agent, state_values):
        super().__init__(agent)
        self._discount = agent.discount
        self._state_values = state_values

    def __call__(self, observation, model):
        del observation
        init_state = model.clone_state()

        def quality(action):
            (observation, reward, _) = yield from model.step(action)
            model.restore_state(init_state)
            # State is the same as observation.
            return reward + self._discount * self._state_values[observation]

        quality_prob_pairs = []
        for action in range(model.action_space.n):
            q = yield from quality(action)
            quality_prob_pairs.append((q, 1 / model.action_space.n))
        return quality_prob_pairs


def make_one_level_binary_tree(
    left_value, right_value, left_reward=0, right_reward=0
):
    """Makes a TabularEnv and new_leaf_rater_class for a 1-level binary tree."""
    # 0, action 0 -> 1 (left)
    # 0, action 1 -> 2 (right)
    (root_state, left_state, right_state) = (0, 1, 2)
    env = testing.TabularEnv(
        init_state=root_state,
        n_actions=2,
        transitions={
            # state: {action: (state', reward, done)}
            root_state: {
                0: (left_state, left_reward, False),
                1: (right_state, right_reward, False),
            },
            # Dummy terminal states, made so we can expand left and right.
            left_state: {0: (3, 0, True), 1: (4, 0, True)},
            right_state: {0: (5, 0, True), 1: (6, 0, True)},
        }
    )
    new_leaf_rater_class = functools.partial(
        TabularNewLeafRater,
        state_values={
            root_state: 0,
            left_state: left_value,
            right_state: right_value,
            # Dummy terminal states.
            3: 0, 4: 0, 5: 0, 6: 0,
        },
    )
    return (env, new_leaf_rater_class)


@pytest.mark.parametrize(
    'left_value,right_value,left_reward,right_reward,expected_action', [
        (1, 0, 0, 0, 0),  # Should choose left because of high value.
        (0, 1, 0, 0, 1),  # Should choose right because of high value.
        (0, 0, 1, 0, 0),  # Should choose left because of high reward.
        (0, 0, 0, 1, 1),  # Should choose right because of high reward.
    ]
)
def test_decision_after_two_passes(
    left_value,
    right_value,
    left_reward,
    right_reward,
    expected_action,
):
    # 0, action 0 -> 1 (left)
    # 0, action 1 -> 2 (right)
    # 2 passes, should choose depending on qualities.
    (env, new_leaf_rater_class) = make_one_level_binary_tree(
        left_value, right_value, left_reward, right_reward
    )
    agent = agents.StochasticMCTSAgent(
        n_passes=2,
        new_leaf_rater_class=new_leaf_rater_class,
    )
    observation = env.reset()
    testing.run_without_suspensions(agent.reset(env, observation))
    (actual_action, _) = testing.run_without_suspensions(agent.act(observation))
    assert actual_action == expected_action


def test_stops_on_done():
    # 0 -> 1 (done)
    # 2 passes, env is not stepped from 1.
    env = testing.TabularEnv(
        init_state=0,
        n_actions=1,
        transitions={0: {0: (1, 0, True)}},
    )
    agent = agents.StochasticMCTSAgent(
        n_passes=2,
        new_leaf_rater_class=functools.partial(
            TabularNewLeafRater,
            state_values={0: 0, 1: 0},
        ),
    )
    observation = env.reset()
    testing.run_without_suspensions(agent.reset(env, observation))
    # rate_new_leaves_fn errors out when rating nodes not in the value table.
    testing.run_without_suspensions(agent.act(observation))


def test_backtracks_because_of_value():
    # 0, action 0 -> 1 (medium value)
    # 0, action 1 -> 2 (high value)
    # 2, action 0 -> 3 (very low value)
    # 2, action 1 -> 3 (very low value)
    # 4 passes, should choose 0.
    env = testing.TabularEnv(
        init_state=0,
        n_actions=2,
        transitions={
            # Root.
            0: {0: (1, 0, False), 1: (2, 0, False)},
            # Left branch, ending here.
            1: {0: (3, 0, True), 1: (4, 0, True)},
            # Right branch, one more level.
            2: {0: (5, 0, False), 1: (6, 0, False)},
            # End of the right branch.
            5: {0: (7, 0, True), 1: (8, 0, True)},
            6: {0: (9, 0, True), 1: (10, 0, True)},
        },
    )
    agent = agents.StochasticMCTSAgent(
        n_passes=4,
        new_leaf_rater_class=functools.partial(
            TabularNewLeafRater,
            state_values={
                0: 0,
                1: 0,
                2: 1,
                3: 0,
                4: 0,
                5: -10,
                6: -10,
            },
        ),
    )
    observation = env.reset()
    testing.run_without_suspensions(agent.reset(env, observation))
    (action, _) = testing.run_without_suspensions(agent.act(observation))
    assert action == 0


def test_backtracks_because_of_reward():
    # 0, action 0 -> 1 (high value, very low reward)
    # 0, action 1 -> 2 (medium value)
    # 2 passes, should choose 1.
    (env, new_leaf_rater_class) = make_one_level_binary_tree(
        left_value=1, left_reward=-10, right_value=0, right_reward=0
    )
    agent = agents.StochasticMCTSAgent(
        n_passes=2,
        new_leaf_rater_class=new_leaf_rater_class,
    )
    observation = env.reset()
    testing.run_without_suspensions(agent.reset(env, observation))
    (action, _) = testing.run_without_suspensions(agent.act(observation))
    assert action == 1


def test_callbacks_are_called():
    mock_callback = mock.MagicMock()
    mock_callback_class = lambda _: mock_callback

    (env, new_leaf_rater_class) = make_one_level_binary_tree(
        left_value=0, right_value=0
    )
    agent = agents.StochasticMCTSAgent(
        n_passes=2,
        new_leaf_rater_class=new_leaf_rater_class,
        callback_classes=[mock_callback_class],
    )
    testing.run_without_suspensions(agent.solve(env))

    mock_callback.on_episode_begin.assert_called_once()
    assert mock_callback.on_pass_begin.call_count == 4
    assert mock_callback.on_model_step.call_count == 4
    assert mock_callback.on_pass_end.call_count == 4
    assert mock_callback.on_real_step.call_count == 2
    mock_callback.on_episode_end.assert_called_once()
