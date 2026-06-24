# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utility functions for Astropy table manipulation.

This module provides helper functions for cleaning and
pre-processing table-like data structures.
"""

import numpy as np
from astropy.table import Table

__all__ = [
    "pad_list_to_length",
    "remove_nan_rows",
]


def pad_list_to_length(length, input_list):
    """
    Pad a list to a given length using NaN values.

    If the input list is shorter than the target length,
    it is extended with `numpy.nan`. If it is longer,
    a ValueError is raised.

    Parameters
    ----------
    length : int
        Target length of the output list.
    input_list : list
        Input list to be padded.

    Returns
    -------
    list
        Padded list of length `length`.

    Raises
    ------
    ValueError
        If `input_list` is longer than `length`.
    """
    diff = length - len(input_list)

    if diff < 0:
        raise ValueError(
            "Input list is longer than the requested target length."
        )

    return input_list + [np.nan] * diff


def remove_nan_rows(table):
    """
    Remove rows containing NaN values from an Astropy Table.

    This function scans all floating-point columns and removes
    any row that contains at least one NaN value.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Input table.

    Returns
    -------
    `~astropy.table.Table`
        Filtered table without NaN-containing rows.
    """
    mask = np.zeros(len(table), dtype=bool)

    for col in table.itercols():
        if col.info.dtype.kind == "f":
            mask |= np.isnan(col)

    return table[~mask]