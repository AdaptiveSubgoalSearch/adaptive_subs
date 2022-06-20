"""Tests for alpacka.agents.max_ent_tree_search."""

import functools

import pytest

from alpacka import envs
from alpacka import testing
from alpacka.agents import mets
from alpacka.agents import stochastic_mcts


# Override the default rater to a fast one.
default_rater = stochastic_mcts.ValueNetworkNewLeafRater  # pylint: disable=invalid-name

all_metses = []

# New leaf raters:
all_metses += [
    functools.partial(mets.MaxEntTreeSearchAgent, new_leaf_rater_class=rater)
    for rater in [
        functools.partial(
            stochastic_mcts.RolloutNewLeafRater,
            rollout_time_limit=2,
        ),
        stochastic_mcts.ValueNetworkNewLeafRater,
        functools.partial(
            stochastic_mcts.QualityNetworkNewLeafRater,
            use_policy=False,
        ),
        functools.partial(
            mets.SoftQualityNetworkNewLeafRater,
            inject_log_temperature=False,
        ),
        functools.partial(
            mets.SoftQualityNetworkNewLeafRater,
            inject_log_temperature=True,
        ),
    ]
]

# Temperature tuners:
all_metses += [
    functools.partial(
        mets.MaxEntTreeSearchAgent,
        temperature_tuner_class=tuner,
        new_leaf_rater_class=default_rater,
    )
    for tuner in [
        mets.ConstantTuner,
        mets.MeanEntropyTuner,
        mets.EntropyRangeTuner,
        mets.StandardDeviationTuner,
    ]
]

# Soft iteration modes:
all_metses += [
    functools.partial(
        mets.MaxEntTreeSearchAgent,
        soft_iteration_class=soft_iteration,
        new_leaf_rater_class=default_rater,
    )
    for soft_iteration in [
        mets.SoftPolicyIteration,
        mets.SoftQIteration,
    ]
]

# Quality accumulation modes:
all_metses += [
    functools.partial(
        mets.MaxEntTreeSearchAgent,
        quality_accumulation=q_acc,
        new_leaf_rater_class=default_rater,
    )
    for q_acc in mets.QualityAccumulation
]

# Quality recalculation modes:
all_metses += [
    functools.partial(
        mets.MaxEntTreeSearchAgent,
        quality_recalculation=q_recalc,
        new_leaf_rater_class=default_rater,
    )
    for q_recalc in mets.QualityRecalculation
]


@pytest.mark.parametrize('agent_fn', all_metses)
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
