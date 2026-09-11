# Licensed under a 3-clause BSD style license - see LICENSE
"""Input/output utilities."""

import logging
from pathlib import Path

import yaml

__all__ = [
    "mkdir_sub_directory",
    "read_yaml",
]

log = logging.getLogger(__name__)


def mkdir_sub_directory(parent_directory, child_directory=None):
    """Create a parent directory and, optionally, a child directory.

    Parameters
    ----------
    parent_directory : str or `~pathlib.Path`
        Parent directory path.
    child_directory : str, optional
        Child directory name.

    Returns
    -------
    path : `~pathlib.Path` or tuple of `~pathlib.Path`
        Parent directory path, or a tuple containing the parent and child
        directory paths when ``child_directory`` is provided.

    Examples
    --------
    >>> parent = mkdir_sub_directory("data")
    >>> parent, child = mkdir_sub_directory("data", "subfolder")
    """
    path_parent = Path(parent_directory)
    path_parent.mkdir(parents=True, exist_ok=True)
    log.info("Directory '%s' created.", path_parent)

    if child_directory is None:
        return path_parent

    path_child = path_parent / child_directory
    path_child.mkdir(parents=True, exist_ok=True)
    log.info("Directory '%s' created.", path_child)

    return path_parent, path_child


def read_yaml(file_path):
    """Read a YAML file.

    Parameters
    ----------
    file_path : str or `~pathlib.Path`
        YAML file path.

    Returns
    -------
    data : object
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    yaml.YAMLError
        If the YAML content cannot be parsed.

    Examples
    --------
    >>> data = read_yaml("data/info.yaml")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} not found.")

    try:
        with file_path.open(encoding="utf-8") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError:
        log.exception("Error parsing YAML file '%s'.", file_path)
        raise
