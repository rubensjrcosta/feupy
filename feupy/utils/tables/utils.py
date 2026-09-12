# Licensed under a 3-clause BSD style license - see LICENSE
"""Utility functions for Astropy table manipulation."""

import numpy as np

__all__ = [
    "pad_list_to_length",
    "remove_nan_rows",
]


def pad_list_to_length(length, input_list):
    """Pad a list to a given length using NaN values.

    Parameters
    ----------
    length : int
        Target length.
    input_list : list
        Input list to pad.

    Returns
    -------
    list
        Padded list of the requested length.

    Raises
    ------
    ValueError
        If the input list is longer than the requested length.
    """
    diff = length - len(input_list)

    if diff < 0:
        raise ValueError("Input list is longer than the requested target length.")

    return input_list + [np.nan] * diff


def remove_nan_rows(table):
    """Remove rows containing NaN values from floating-point columns.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Input table.

    Returns
    -------
    table : `~astropy.table.Table`
        Table without rows containing NaN values in floating-point columns.
    """
    mask = np.zeros(len(table), dtype=bool)

    for column in table.itercols():
        if column.info.dtype.kind == "f":
            mask |= np.isnan(column)

    return table[~mask]
