# Licensed under a 3-clause BSD style license - see LICENSE

import pytest
import yaml

from feupy.utils import io
from feupy.utils.io import mkdir_sub_directory, read_yaml


def test_all():
    expected = {
        "mkdir_sub_directory",
        "read_yaml",
    }

    assert set(io.__all__) == expected

    for name in io.__all__:
        assert hasattr(io, name)


def test_mkdir_sub_directory_parent(tmp_path):
    parent = tmp_path / "data"

    result = mkdir_sub_directory(parent)

    assert result == parent
    assert result.is_dir()


def test_mkdir_sub_directory_parent_and_child(tmp_path):
    parent = tmp_path / "data"

    result_parent, result_child = mkdir_sub_directory(parent, "subfolder")

    assert result_parent == parent
    assert result_parent.is_dir()
    assert result_child == parent / "subfolder"
    assert result_child.is_dir()


def test_mkdir_sub_directory_existing(tmp_path):
    parent = tmp_path / "data"
    parent.mkdir()

    result = mkdir_sub_directory(parent)

    assert result == parent
    assert result.is_dir()


def test_read_yaml(tmp_path):
    filename = tmp_path / "config.yaml"
    filename.write_text(
        "name: Crab\nvalue: 42\nenabled: true\n",
        encoding="utf-8",
    )

    result = read_yaml(filename)

    assert result == {
        "name": "Crab",
        "value": 42,
        "enabled": True,
    }


def test_read_yaml_accepts_string_path(tmp_path):
    filename = tmp_path / "config.yaml"
    filename.write_text("value: 1\n", encoding="utf-8")

    result = read_yaml(str(filename))

    assert result == {"value": 1}


def test_read_yaml_missing_file(tmp_path):
    filename = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="not found"):
        read_yaml(filename)


def test_read_yaml_invalid_content(tmp_path):
    filename = tmp_path / "invalid.yaml"
    filename.write_text("value: [1, 2\n", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        read_yaml(filename)
