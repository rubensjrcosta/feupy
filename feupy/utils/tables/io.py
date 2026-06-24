# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Input/output utilities for Astropy tables.

This module provides functions to read and write tables
in FITS and ECSV formats.
"""

import os
from astropy.table import Table

__all__ = [
    "write_table",
    "read_table",
]


def _get_format(file_name):
    """
    Infer Astropy format from file extension.

    Parameters
    ----------
    file_name : str
        File name.

    Returns
    -------
    str
        Astropy table format string.

    Raises
    ------
    ValueError
        If extension is not supported.
    """
    if file_name.endswith(".fits"):
        return "fits"
    elif file_name.endswith(".csv"):
        return "ascii.ecsv"
    else:
        raise ValueError("File must end with .fits or .csv")


def write_table(table, path_file, file_name, overwrite=False):
    """
    Write an Astropy Table to disk.

    Supports FITS and ECSV (CSV-like) formats.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table to write.
    path_file : str
        Output directory.
    file_name : str
        File name (must end in .fits or .csv).
    overwrite : bool, optional
        Overwrite existing file.

    Returns
    -------
    None
    """
    fmt = _get_format(file_name)
    path = os.path.abspath(os.path.join(path_file, file_name))

    table.write(path, format=fmt, overwrite=overwrite)


def read_table(path_file, file_name):
    """
    Read an Astropy Table from disk.

    Supports FITS and ECSV formats.

    Parameters
    ----------
    path_file : str
        Directory containing file.
    file_name : str
        File name (must end in .fits or .csv).

    Returns
    -------
    `~astropy.table.Table`
        Loaded table.
    """
    fmt = _get_format(file_name)
    path = os.path.abspath(os.path.join(path_file, file_name))

    return Table.read(path, format=fmt)