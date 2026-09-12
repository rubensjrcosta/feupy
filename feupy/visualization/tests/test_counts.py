# Licensed under a 3-clause BSD style license - see LICENSE.rst

from unittest.mock import MagicMock, patch

from astropy.table import Table

from feupy.visualization.counts import show_hist_counts


def make_counts_table():
    table = Table()
    table["counts"] = [10, 20, 30]
    table["counts_off"] = [5, 10, 15]
    table["excess"] = [5, 10, 15]
    table["sqrt_ts"] = [2.0, 3.0, 4.0]

    return table


def test_show_hist_counts():
    table = make_counts_table()

    with patch("feupy.visualization.counts.plt.subplots") as subplots:
        axes = [MagicMock() for _ in range(4)]
        subplots.return_value = ("fig", axes)

        result = show_hist_counts(table)

    assert result is None

    subplots.assert_called_once_with(
        1,
        4,
        figsize=(12, 4),
    )

    expected = [
        ("counts", "Counts"),
        ("counts_off", "Counts Off"),
        ("excess", "Excess"),
        ("sqrt_ts", r"Significance ($\sigma$)"),
    ]

    for ax, (column, xlabel) in zip(axes, expected, strict=True):
        ax.hist.assert_called_once_with(table[column])
        ax.set_xlabel.assert_called_once_with(xlabel)

    axes[0].set_ylabel.assert_called_once_with("Frequency")


def test_show_hist_counts_save():
    table = make_counts_table()

    with (
        patch("feupy.visualization.counts.plt.subplots") as subplots,
        patch("feupy.visualization.counts.plt.savefig") as savefig,
    ):
        axes = [MagicMock() for _ in range(4)]
        subplots.return_value = ("fig", axes)

        show_hist_counts(
            table,
            file_path="counts.png",
        )

    savefig.assert_called_once_with(
        "counts.png",
        bbox_inches="tight",
    )
