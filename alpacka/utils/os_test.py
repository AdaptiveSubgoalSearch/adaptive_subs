"""Tests for alpacka.utils.os."""

import os

import pytest

from alpacka.utils import os as alpacka_os


def _write(path, content):
    with open(path, 'w') as f:
        f.write(content)


def _assert_content(path, content):
    with open(path, 'r') as f:
        assert f.read() == content


def test_single_write_file(tmp_path):
    path = tmp_path / 'tmp'
    with alpacka_os.atomic_dump((path,)) as (dump_path,):
        _write(dump_path, 'foo')

    _assert_content(path, 'foo')


def test_single_write_directory(tmp_path):
    dir_path = tmp_path / 'tmp'

    with alpacka_os.atomic_dump((dir_path,)) as (dump_dir_path,):
        os.mkdir(dir_path)
        _write(dump_dir_path / 'file', 'foo')

    _assert_content(dir_path / 'file', 'foo')


def test_double_write_file(tmp_path):
    path = tmp_path / 'tmp'

    # Write the first time.
    _write(path, 'foo')

    # Write the second time.
    with alpacka_os.atomic_dump((path,)) as (dump_path,):
        _write(dump_path, 'bar')

    # The content should be overwritten.
    _assert_content(path, 'bar')


def test_double_write_directory(tmp_path):
    dir_path = tmp_path / 'tmp'

    # Write the first time.
    os.mkdir(dir_path)
    _write(dir_path / 'file', 'foo')

    # Write the second time.
    with alpacka_os.atomic_dump((dir_path,)) as (dump_dir_path,):
        os.mkdir(dump_dir_path)
        _write(dump_dir_path / 'file', 'bar')

    # The content should be overwritten.
    _assert_content(dir_path / 'file', 'bar')


def test_partial_write_file(tmp_path):
    path = tmp_path / 'tmp'

    # Write the first time.
    _write(path, 'foo')

    # Write the second time, with interruption.
    with pytest.raises(Exception):
        with alpacka_os.atomic_dump((path,)) as (dump_path,):
            _write(dump_path, 'bar')
            raise Exception

    # The content should not be overwritten.
    _assert_content(path, 'foo')


def test_partial_write_directory(tmp_path):
    dir_path = tmp_path / 'tmp'

    # Write the first time.
    os.mkdir(dir_path)
    _write(dir_path / 'file', 'foo')

    # Write the second time, with interruption.
    with pytest.raises(Exception):
        with alpacka_os.atomic_dump((dir_path,)) as (dump_dir_path,):
            os.mkdir(dump_dir_path)
            _write(dump_dir_path / 'file', 'bar')
            raise Exception

    # The content should not be overwritten.
    _assert_content(dir_path / 'file', 'foo')


def test_write_to_multiple_files(tmp_path):
    path1 = tmp_path / 'tmp1'
    path2 = tmp_path / 'tmp2'
    with alpacka_os.atomic_dump((path1, path2)) as (dump_path1, dump_path2):
        _write(dump_path1, 'foo')
        _write(dump_path2, 'bar')

    _assert_content(path1, 'foo')
    _assert_content(path2, 'bar')
