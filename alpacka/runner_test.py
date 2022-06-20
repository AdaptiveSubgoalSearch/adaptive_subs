"""Tests for alpacka.runner."""

import functools

import gin

from alpacka import runner
from alpacka import runner_callbacks


def test_smoke(tmpdir, capsys):
    n_epochs = 3
    runner.Runner(
        output_dir=tmpdir,
        n_envs=2,
        n_epochs=n_epochs,
    ).run()

    # Check that metrics were printed in each epoch.
    captured = capsys.readouterr()
    assert captured.out.count('return_mean') == n_epochs


def test_eval(tmpdir, capsys):
    n_epochs = 3
    runner.Runner(
        output_dir=tmpdir,
        n_envs=2,
        n_epochs=n_epochs,
        callback_classes=(functools.partial(
            runner_callbacks.EvaluationCallback, eval_period=1
        ),),
    ).run()

    # Check that eval metrics were printed in each epoch.
    captured = capsys.readouterr()
    assert captured.out.count('eval_episode/return_mean') == n_epochs


def test_metric_smoothing(tmpdir, capsys):
    n_epochs = 3
    runner.Runner(
        output_dir=tmpdir,
        n_envs=2,
        n_epochs=n_epochs,
        metric_smoothing=('return_mean', 0.9),
    ).run()

    # Check that the smoothed metrics are printed.
    captured = capsys.readouterr()
    assert captured.out.count('return_mean/smoothing_0.9') == n_epochs


def test_restarts(tmpdir, capsys):
    runner.Runner(
        output_dir=tmpdir,
        n_envs=2,
        n_epochs=1,
    ).run()

    # Check that one epoch has been run.
    captured = capsys.readouterr()
    assert captured.out.count('0 |') > 0
    assert captured.out.count('1 |') == 0

    runner.Runner(
        output_dir=tmpdir,
        n_envs=2,
        n_epochs=2,
    ).run()

    # Check that another epoch has been run.
    captured = capsys.readouterr()
    assert captured.out.count('0 |') == 0
    assert captured.out.count('1 |') > 0
    assert captured.out.count('2 |') == 0


@gin.configurable
class _TestCallback(runner_callbacks.RunnerCallback):

    def __init__(self, runner, epochs_and_foos, foo=gin.REQUIRED):
        super().__init__(runner)
        self.epochs_and_foos = epochs_and_foos
        self.foo = foo

    def on_epoch_end(self, epoch, network_params):
        del network_params
        self.epochs_and_foos.append((epoch, self.foo))


def test_reconfigation(tmpdir):
    gin.parse_config("""
        _TestCallback.foo = None
        scope_x/_TestCallback.foo = 'x'
        scope_y/_TestCallback.foo = 'y'
        scope_z/_TestCallback.foo = 'z'
    """)

    epochs_and_foos = []
    runner.Runner(
        output_dir=tmpdir,
        n_envs=2,
        n_epochs=8,
        callback_classes=(
            functools.partial(
                runner_callbacks.ReconfigurationCallback, epoch_to_scope={
                    1: 'scope_x',
                    5: 'scope_y',
                    7: 'scope_z',
                }
            ),
            functools.partial(_TestCallback, epochs_and_foos=epochs_and_foos),
        ),
    ).run()

    (x, y, z) = ('x', 'y', 'z')
    assert epochs_and_foos == list(enumerate([None, x, x, x, x, y, y, z]))
