"""Tests for alpacka.utils.compression."""

import os

from alpacka.utils import compression


def test_dumps_and_loads(tmpdir):
    x = (1, 2, 3.45, [6, 7, 8], {9, 0}, 'test', {'abc': 'def'})
    path = os.path.join(tmpdir, 'test')

    compression.dump(x, path)
    y = compression.load(path)

    assert x == y
