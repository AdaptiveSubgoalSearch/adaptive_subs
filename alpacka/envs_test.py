"""Tests for alpacka.envs."""

import functools
import warnings

import flaky
import numpy as np
import pytest

from alpacka import envs
from alpacka.utils import space


envs_to_test = envs.native_envs
if envs.Atari.installed:
    envs_to_test.append(envs.Atari)
if envs.GoogleFootball.installed:
    envs_to_test.append(envs.GoogleFootball)
if envs.Sokoban.installed:
    envs_to_test.append(envs.Sokoban)

wrappers_to_test = [
    functools.partial(envs.TimeLimitWrapper, max_episode_steps=10),
    functools.partial(envs.FrameStackWrapper),
]


def env_variants(env_class):
    stochasticities = getattr(env_class, 'allowed_stochasticities', None)
    if stochasticities is None:
        return [env_class]
    else:
        return [
            functools.partial(env_class, stochasticity=stochasticity)
            for stochasticity in stochasticities
        ]


envs_to_test = [
    variant for env_class in envs_to_test for variant in env_variants(env_class)
]

envs_to_test += [
    functools.partial(wrapper, envs.CartPole()) for wrapper in wrappers_to_test
]


@pytest.fixture(scope='module', params=envs_to_test)
def env_fn(request):
    return request.param


def test_reset_and_step_observation_shapes(env_fn):
    # Set up
    env = env_fn()

    # Run
    obs_reset = env.reset()
    (obs_step, _, _, _) = env.step(env.action_space.sample())

    # Test
    assert obs_reset.shape == env.observation_space.shape
    assert obs_step.shape == env.observation_space.shape


def test_restore_after_reset(env_fn):
    # Set up
    env = env_fn()
    obs = env.reset()
    state = env.clone_state()

    # Run
    env.reset()
    obs_ = env.restore_state(state)
    state_ = env.clone_state()

    # Test
    env.step(env.action_space.sample())  # Test if can take step
    np.testing.assert_array_equal(obs, obs_)
    np.testing.assert_equal(state, state_)


def test_restore_after_step(env_fn):
    # Set up
    env = env_fn()
    obs = env.reset()
    state = env.clone_state()

    # Run
    env.step(env.action_space.sample())
    obs_ = env.restore_state(state)
    state_ = env.clone_state()

    # Test
    env.step(env.action_space.sample())  # Test if can take step
    np.testing.assert_array_equal(obs, obs_)
    np.testing.assert_equal(state, state_)


def test_restore_to_step_after_reset(env_fn):
    # Set up
    env = env_fn()
    # Try making a step until it's not "done".
    done = True
    while done:
        env.reset()
        (obs, _, done, _) = env.step(env.action_space.sample())
    state = env.clone_state()

    # Run
    env.reset()
    obs_ = env.restore_state(state)
    state_ = env.clone_state()

    # Test
    env.step(env.action_space.sample())  # Test if can take step
    np.testing.assert_array_equal(obs, obs_)
    np.testing.assert_equal(state, state_)


def test_restore_in_place_of_reset(env_fn):
    # Set up
    env_rolemodel, env_imitator = env_fn(), env_fn()
    obs = env_rolemodel.reset()
    state = env_rolemodel.clone_state()

    # Run
    obs_ = env_imitator.restore_state(state)
    state_ = env_imitator.clone_state()

    # Test
    env_imitator.step(env_imitator.action_space.sample())
    np.testing.assert_array_equal(obs, obs_)
    np.testing.assert_equal(state, state_)


def _test_determinism(env):
    # Sample 2 consectutive actions.
    action1 = env.action_space.sample()
    action2 = env.action_space.sample()

    # First episode.
    env.reset()
    step1 = env.step(action1)
    step2 = env.step(action2)

    # Second episode, starting with a reset - should be exactly the same.
    env.reset()
    np.testing.assert_equal(env.step(action1), step1)
    np.testing.assert_equal(env.step(action2), step2)


def _assert_exists_stochastic_action_sequence(
    env, init_state, seq_len=10, n_reps=10
):
    stochastic = False

    # Go over all actions. Some of them should be stochastic, i.e. yield
    # different observations after making the same sequence of actions.
    for init_action in space.element_iter(env.action_space):
        action_seq = [init_action] + [
            env.action_space.sample() for _ in range(seq_len)
        ]
        observations = set()

        for _ in range(n_reps):
            if init_state is None:
                env.reset()
            else:
                env.restore_state(init_state)

            for action in action_seq:
                (obs, rew, done, _) = env.step(action)
                if done:
                    break

            # Convert to string to make hashable.
            observations.add(str(obs.tobytes()) + str(rew))

            if len(observations) > 1:
                stochastic = True
                break

        if stochastic:
            break

    assert stochastic


def _test_stochasticity_episodic(env):
    # Sample 2 consectutive actions.
    action1 = env.action_space.sample()
    action2 = env.action_space.sample()

    # First episode.
    env.reset()
    state = env.clone_state()
    step1 = env.step(action1)
    step2 = env.step(action2)

    # Second episode, starting from a restored state - should be exactly the
    # same.
    env.restore_state(state)
    np.testing.assert_equal(env.step(action1), step1)
    np.testing.assert_equal(env.step(action2), step2)

    # Episodes starting with a reset should be different.
    _assert_exists_stochastic_action_sequence(env, init_state=None)


def _test_stochasticity_universal(env):
    # Mark an initial state.
    env.reset()
    state = env.clone_state()

    # Episodes starting from that (or any other) state should be different.
    _assert_exists_stochastic_action_sequence(env, state)


@flaky.flaky
def test_stochasticity(env_fn):
    env = env_fn()
    testers = {
        envs.Stochasticity.none: _test_determinism,
        envs.Stochasticity.episodic: _test_stochasticity_episodic,
        envs.Stochasticity.universal: _test_stochasticity_universal,
        envs.Stochasticity.unknown: lambda _: warnings.warn(
            'Stochasticity mode not specified for environment {}.'.format(env)
        )
    }
    testers[env.stochasticity](env)
