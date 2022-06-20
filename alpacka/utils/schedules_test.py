"""Test parameter schedules."""

import pytest

from alpacka import testing
from alpacka.agents import core
from alpacka.utils import schedules


mock_env = testing.mock_env_fixture


@pytest.mark.parametrize('schedule,expected_points', [
    (
        schedules.LinearAnnealing(max_value=10, min_value=0, n_epochs=10),
        enumerate(range(10, 0, -1))
    ),
    (schedules.RsqrtAnnealing(scale=1), [(0, 1), (3, 0.5), (15, 0.25)]),
    (schedules.RsqrtAnnealing(scale=3), [(0, 1), (9, 0.5), (45, 0.25)]),
    (
        schedules.RsqrtAnnealing(scale=1, max_value=2),
        [(0, 2), (3, 1.0), (15, 0.5)]
    ),
])
def test_schedule(schedule, expected_points, mock_env):
    # Set up
    attr_name = 'pied_piper'
    agent = core.RandomAgent(parameter_schedules={attr_name: schedule})

    # Run & Test
    for (epoch, x_value) in expected_points:
        testing.run_without_suspensions(agent.solve(mock_env, epoch=epoch))
        assert getattr(agent, attr_name) == x_value
