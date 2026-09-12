# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Histogram plotting utilities for count statistics."""

import matplotlib.pyplot as plt

__all__ = ["show_hist_counts"]


def show_hist_counts(table, file_path=None):
    """Plot histograms for count-related statistics.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table containing the ``counts``, ``counts_off``, ``excess``,
        and ``sqrt_ts`` columns.
    file_path : str or `~pathlib.Path`, optional
        Destination path used to save the figure.

    Returns
    -------
    None
    """
    _, axes = plt.subplots(
        1,
        4,
        figsize=(12, 4),
    )

    columns = [
        ("counts", "Counts"),
        ("counts_off", "Counts Off"),
        ("excess", "Excess"),
        ("sqrt_ts", r"Significance ($\sigma$)"),
    ]

    for ax, (column, xlabel) in zip(axes, columns, strict=True):
        ax.hist(table[column])
        ax.set_xlabel(xlabel)

    axes[0].set_ylabel("Frequency")

    if file_path:
        plt.savefig(
            file_path,
            bbox_inches="tight",
        )

    return None
