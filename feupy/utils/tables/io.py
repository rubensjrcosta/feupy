# Licensed under a 3-clause BSD style license - see LICENSE
"""Input/output utilities for Astropy tables."""

from pathlib import Path

from astropy.table import Table

__all__ = [
    "write_table",
    "read_table",
]


def _get_format(file_name):
    """Infer the Astropy table format from a file extension.

    Parameters
    ----------
    file_name : str or `~pathlib.Path`
        File name.

    Returns
    -------
    format : str
        Astropy table format.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    suffix = Path(file_name).suffix.lower()

    if suffix == ".fits":
        return "fits"
    if suffix == ".csv":
        return "ascii.ecsv"

    raise ValueError("File must end with .fits or .csv.")


def write_table(table, path_file, file_name, overwrite=False):
    """Write an Astropy table to disk.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table to write.
    path_file : str or `~pathlib.Path`
        Output directory.
    file_name : str or `~pathlib.Path`
        File name ending in ``.fits`` or ``.csv``.
    overwrite : bool, optional
        Whether to overwrite an existing file. Default is False.
    """
    fmt = _get_format(file_name)
    path = Path(path_file) / file_name
    table.write(path, format=fmt, overwrite=overwrite)


def read_table(path_file, file_name):
    """Read an Astropy table from disk.

    Parameters
    ----------
    path_file : str or `~pathlib.Path`
        Directory containing the file.
    file_name : str or `~pathlib.Path`
        File name ending in ``.fits`` or ``.csv``.

    Returns
    -------
    table : `~astropy.table.Table`
        Loaded table.
    """
    fmt = _get_format(file_name)
    path = Path(path_file) / file_name
    return Table.read(path, format=fmt)
