"""Tests for alpacka.envs.wrappers."""

import numpy as np
import pytest

from alpacka import envs
from alpacka.envs import cartpole
from alpacka.envs import wrappers


def test_cached_equals_uncached_cartpole():
    def create_uncached():
        return wrappers.TimeLimitWrapper(
            cartpole.CartPole(), max_episode_steps=10
        )

    uncached_env = create_uncached()
    cached_env = wrappers.StateCachingWrapper(create_uncached())
    assert cached_env.observation_space == uncached_env.observation_space
    assert cached_env.action_space == uncached_env.action_space

    # Reset the cached env, then clone the state into the uncached env.
    # The other way around wouldn't work, because the state representations
    # are different and we can't bypass the StateCachingWrapper when restoring.
    cached_init_obs = cached_env.reset()
    uncached_init_state = cached_env.env.clone_state()
    uncached_init_obs = uncached_env.restore_state(uncached_init_state)
    np.testing.assert_equal(cached_init_obs, uncached_init_obs)
    cached_init_state = cached_env.clone_state()

    # Run a random sequence of actions and compare the transitions.
    uncached_done = False
    uncached_transitions = []
    actions = []
    while not uncached_done:
        action = uncached_env.action_space.sample()
        actions.append(action)

        uncached_transition = uncached_env.step(action)
        cached_transition = cached_env.step(action)
        np.testing.assert_equal(cached_transition, uncached_transition)
        uncached_transitions.append(uncached_transition)

        (_, _, uncached_done, _) = uncached_transition

    # Rewind to the initial state and check again.
    cached_init_obs = cached_env.restore_state(cached_init_state)
    np.testing.assert_equal(cached_init_obs, uncached_init_obs)

    for (action, uncached_transition) in zip(actions, uncached_transitions):
        cached_transition = cached_env.step(action)
        np.testing.assert_equal(cached_transition, uncached_transition)


@pytest.fixture(params=[1, 4])
def n_frames(request):
    return request.param


@pytest.fixture(params=[False, True])
def concatenate(request):
    return request.param


def test_frame_stack_wrapper_obs_space_shape(n_frames, concatenate):
    env = envs.Atari()
    wrapped_env = envs.FrameStackWrapper(env, n_frames=n_frames,
                                         axis=-1, concatenate=concatenate)

    if concatenate:
        shape = list(env.observation_space.shape)
        shape[-1] = env.observation_space.shape[-1] * n_frames
        assert tuple(shape) == wrapped_env.observation_space.shape
    else:
        assert env.observation_space.shape + (n_frames,) == \
               wrapped_env.observation_space.shape


def test_frame_stack_wrapper_frame_order(n_frames, concatenate):
    env = envs.Atari(stochasticity=envs.Stochasticity.none)
    wrapped_env = envs.FrameStackWrapper(env, n_frames=n_frames,
                                         axis=-1, concatenate=concatenate)

    actions = [env.action_space.sample() for _ in range(10)]

    env_obs = [env.reset()]
    for action in actions:
        obs, _, _, _ = env.step(action)
        env_obs.append(obs)

    wrapped_env_obs = [wrapped_env.reset()]
    for action in actions:
        obs, _, _, _ = wrapped_env.step(action)
        wrapped_env_obs.append(obs)

    for i, stacked_frame in enumerate(wrapped_env_obs):
        frames_to_stack = env_obs[max(i + 1 - n_frames, 0): i + 1]
        frames_to_stack = (n_frames - len(frames_to_stack)) * \
                          [np.zeros(env.observation_space.shape)] + \
                          frames_to_stack

        if concatenate:
            assert np.array_equal(np.concatenate(frames_to_stack, axis=-1),
                                  stacked_frame)
        else:
            assert np.array_equal(np.stack(frames_to_stack, axis=-1),
                                  stacked_frame)
