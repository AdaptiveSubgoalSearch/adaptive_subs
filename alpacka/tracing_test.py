"""Tests for alpacka.tracing."""

from unittest import mock

from alpacka import tracing


class Wildcard:

    def __eq__(self, other):
        return True


def test_callback_traces_real_trajectory():
    # Trajectory of length 2.
    callback = tracing.TraceCallback(agent=None)
    mock_env = mock.MagicMock()

    mock_env.state_info = 'state0'
    callback.on_episode_begin(mock_env, 'obs0', epoch=0)
    mock_env.state_info = 'state1'
    callback.on_real_step({'agent_info': 1}, 'action1', 'obs1', 0.1, False)
    mock_env.state_info = 'state2'
    callback.on_real_step({'agent_info': 2}, 'action2', 'obs2', 0.2, True)
    callback.on_episode_end()

    assert callback.trace.trajectory == tracing.Trajectory(
        init_state=tracing.State(
            state_info='state0', node_id=Wildcard(), terminal=False
        ),
        transitions=[
            tracing.RealTransition(
                agent_info={'agent_info': 1},
                action='action1',
                reward=0.1,
                passes=[],
                to_state=tracing.State(
                    state_info='state1', node_id=Wildcard(), terminal=False
                ),
            ),
            tracing.RealTransition(
                agent_info={'agent_info': 2},
                action='action2',
                reward=0.2,
                passes=[],
                to_state=tracing.State(
                    state_info='state2', node_id=Wildcard(), terminal=True
                ),
            )
        ],
    )


def test_callback_traces_single_model_pass():
    # Trajectory of length 1 with a single pass of length 2.
    callback = tracing.TraceCallback(agent=None)
    mock_env = mock.MagicMock()

    callback.on_episode_begin(mock_env, 'obs0', epoch=0)
    callback.on_pass_begin()
    mock_env.state_info = 'state1_1'
    callback.on_model_step(
        {'agent_info': 11}, 'action1_1', 'obs1_1', 0.11, False
    )
    mock_env.state_info = 'state1_2'
    callback.on_model_step(
        {'agent_info': 12}, 'action1_2', 'obs1_2', 0.12, False
    )
    callback.on_pass_end()
    callback.on_real_step({'agent_info': 1}, 'action1', 'obs1', 0.1, True)
    callback.on_episode_end()

    assert callback.trace.trajectory == tracing.Trajectory(
        init_state=Wildcard(),
        transitions=[
            tracing.RealTransition(
                agent_info={'agent_info': 1},
                action='action1',
                reward=0.1,
                passes=[[
                    tracing.ModelTransition(
                        agent_info={'agent_info': 11},
                        action='action1_1',
                        reward=0.11,
                        to_state=tracing.State(
                            state_info='state1_1',
                            node_id=Wildcard(),
                            terminal=False,
                        ),
                    ),
                    tracing.ModelTransition(
                        agent_info={'agent_info': 12},
                        action='action1_2',
                        reward=0.12,
                        to_state=tracing.State(
                            state_info='state1_2',
                            node_id=Wildcard(),
                            terminal=False,
                        ),
                    ),
                ]],
                to_state=Wildcard(),
            ),
        ],
    )


def test_callback_traces_two_model_passes():
    # Trajectory of length 1 with two passes of length 1.
    callback = tracing.TraceCallback(agent=None)
    mock_env = mock.MagicMock()

    callback.on_episode_begin(mock_env, 'obs0', epoch=0)
    callback.on_pass_begin()
    mock_env.state_info = 'state1_1'
    callback.on_model_step(
        {'agent_info': 11}, 'action1_1', 'obs1_1', 0.11, False
    )
    callback.on_pass_end()
    callback.on_pass_begin()
    mock_env.state_info = 'state1_2'
    callback.on_model_step(
        {'agent_info': 12}, 'action1_2', 'obs1_2', 0.12, False
    )
    callback.on_pass_end()
    callback.on_real_step({'agent_info': 1}, 'action1', 'obs1', 0.1, True)
    callback.on_episode_end()

    assert callback.trace.trajectory == tracing.Trajectory(
        init_state=Wildcard(),
        transitions=[
            tracing.RealTransition(
                agent_info={'agent_info': 1},
                action='action1',
                reward=0.1,
                passes=[
                    [tracing.ModelTransition(
                        agent_info={'agent_info': 11},
                        action='action1_1',
                        reward=0.11,
                        to_state=tracing.State(
                            state_info='state1_1',
                            node_id=Wildcard(),
                            terminal=False,
                        ),
                    )],
                    [tracing.ModelTransition(
                        agent_info={'agent_info': 12},
                        action='action1_2',
                        reward=0.12,
                        to_state=tracing.State(
                            state_info='state1_2',
                            node_id=Wildcard(),
                            terminal=False,
                        ),
                    )],
                ],
                to_state=Wildcard(),
            ),
        ],
    )


def test_callback_builds_tree():
    # Trajectory of length 2 with two passes of length 1 from the root.
    callback = tracing.TraceCallback(agent=None)
    mock_env = mock.MagicMock()

    mock_env.state_info = 'state0'
    callback.on_episode_begin(mock_env, 'obs0', epoch=0)
    callback.on_pass_begin()
    mock_env.state_info = 'state1'
    callback.on_model_step({'agent_info': 1}, 'left', 'obs1', 0.1, False)
    callback.on_pass_end()
    callback.on_pass_begin()
    mock_env.state_info = 'state2_1'
    callback.on_model_step({'agent_info': 21}, 'right', 'obs2_1', 0.21, False)
    callback.on_pass_end()
    mock_env.state_info = 'state2_2'
    callback.on_real_step({'agent_info': 22}, 'right', 'obs2_2', 0.22, True)
    callback.on_episode_end()

    tree = callback.trace.tree
    assert tree == [
        tracing.TreeNode(
            agent_info={'agent_info': 22},
            children={
                'left': Wildcard(),
                'right': Wildcard(),
            },
        ),
        tracing.TreeNode(
            agent_info=None,
            children={},
        ),
        tracing.TreeNode(
            agent_info=None,
            children={},
        )
    ]
    root_id = 0
    left_id = tree[root_id].children['left']
    right_id = tree[root_id].children['right']

    assert callback.trace.trajectory == tracing.Trajectory(
        init_state=tracing.State(
            state_info='state0',
            node_id=root_id,
            terminal=False,
        ),
        transitions=[
            tracing.RealTransition(
                agent_info={'agent_info': 22},
                action='right',
                reward=0.22,
                passes=[
                    [tracing.ModelTransition(
                        agent_info={'agent_info': 1},
                        action='left',
                        reward=0.1,
                        to_state=tracing.State(
                            state_info='state1',
                            node_id=left_id,
                            terminal=False,
                        ),
                    )],
                    [tracing.ModelTransition(
                        agent_info={'agent_info': 21},
                        action='right',
                        reward=0.21,
                        to_state=tracing.State(
                            state_info='state2_1',
                            node_id=right_id,
                            terminal=False,
                        ),
                    )],
                ],
                to_state=tracing.State(
                    state_info='state2_2',
                    node_id=right_id,
                    terminal=True,
                ),
            ),
        ],
    )
