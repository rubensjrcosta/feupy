# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Sensitivity plotting utilities."""

import astropy.units as u
import matplotlib.pyplot as plt
from astropy.visualization import quantity_support
from gammapy.maps.axes import UNIT_STRING_FORMAT

from feupy.visualization.utils.labels import (
    DEFAULT_XAXIS_LABEL,
    DEFAULT_YAXIS_LABEL,
)

__all__ = [
    "plot_irfs",
    "plot_irfs_superpose",
    "plot_sensitivity_from_table",
    "plot_tables_sensitivity",
]


def _get_irf_label(table, int_sens_label=True):
    """Build the display label for an IRF sensitivity table."""
    label = table.meta.get("IRF_NAME", "IRF")

    if int_sens_label and "INT_SENS" in table.meta:
        int_sens = u.Quantity(table.meta["INT_SENS"])
        unit = int_sens.unit.to_string(UNIT_STRING_FORMAT)
        label += f" ({int_sens.value:.2e} {unit})"

    return label


def plot_sensitivity_from_table(
    sens_table,
    which="e2dnde",
    ax=None,
    plot_xerr=False,
    **kwargs,
):
    """Plot a sensitivity quantity from a table.

    Parameters
    ----------
    sens_table : `~astropy.table.Table`
        Sensitivity table.
    which : str, optional
        Table column to plot. Default is ``"e2dnde"``.
    ax : `matplotlib.axes.Axes`, optional
        Axes used for plotting. If None, the current axes are used.
    plot_xerr : bool, optional
        Whether to plot energy-bin half widths as x-errors.
    **kwargs : dict
        Additional keyword arguments passed to ``Axes.errorbar``.

    Returns
    -------
    `matplotlib.axes.Axes`
        Axes containing the sensitivity curve.
    """
    ax = plt.gca() if ax is None else ax

    energy = sens_table["e_ref"]
    sensitivity = sens_table[which]

    xlabel = f"Energy [{energy.unit.to_string(UNIT_STRING_FORMAT)}]"
    ylabel = {
        "excess": "Excess counts",
        "background": "Background counts",
        "on_radii": (
            f"On region radius [{sensitivity.unit.to_string(UNIT_STRING_FORMAT)}]"
        ),
        "e2dnde": (
            f"Flux Sensitivity [{sensitivity.unit.to_string(UNIT_STRING_FORMAT)}]"
        ),
    }.get(
        which,
        f"{which} [{sensitivity.unit.to_string(UNIT_STRING_FORMAT)}]",
    )

    xerr = None
    if plot_xerr and "e_min" in sens_table.colnames:
        xerr = (sens_table["e_max"] - sens_table["e_min"]) / 2

    with quantity_support():
        ax.errorbar(
            energy,
            sensitivity,
            xerr=xerr,
            **kwargs,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def plot_irfs(
    tables,
    model=None,
    energy_bounds=None,
    ax=None,
    which="e2dnde",
    which_label="both",
    int_sens_label=True,
):
    """Plot sensitivity curves for a list of IRF tables.

    Parameters
    ----------
    tables : list
        Sensitivity tables.
    model : object, optional
        Model with a ``spectral_model`` attribute.
    energy_bounds : `~astropy.units.Quantity`, optional
        Energy bounds used for the model overlay.
    ax : `matplotlib.axes.Axes`, optional
        Axes used for plotting.
    which : str, optional
        Sensitivity quantity to plot.
    which_label : str, optional
        Retained for API compatibility.
    int_sens_label : bool, optional
        Whether to append integrated sensitivity to labels.

    Returns
    -------
    `matplotlib.axes.Axes`
        Axes containing the plot.
    """
    del which_label

    if not isinstance(tables, list):
        raise TypeError("`tables` must be a list of tables.")

    ax = plt.gca() if ax is None else ax

    ax.set_prop_cycle(
        color=["blue", "red", "green"],
        linestyle=["solid", "solid", "solid"],
    )

    for table in tables:
        label = _get_irf_label(
            table,
            int_sens_label=int_sens_label,
        )

        plot_sensitivity_from_table(
            table,
            which=which,
            ax=ax,
            label=label,
        )

    if model is not None:
        plot_kwargs = {
            "ax": ax,
            "sed_type": "e2dnde",
        }

        model.spectral_model.plot(
            energy_bounds=energy_bounds,
            label=model.name,
            color="k",
            **plot_kwargs,
        )
        model.spectral_model.plot_error(
            energy_bounds=energy_bounds,
            **plot_kwargs,
        )

    ax.legend(
        loc="best",
        fontsize=7,
    )

    return ax


def plot_irfs_superpose(
    tables_south,
    tables_north,
    model=None,
    energy_bounds=None,
    ax=None,
    which="e2dnde",
    which_label="both",
    int_sens_label=True,
):
    """Plot South and North IRF sensitivity curves on the same axes.

    Parameters
    ----------
    tables_south, tables_north : list
        Sensitivity tables for the South and North sites.
    model : object, optional
        Model with a ``spectral_model`` attribute.
    energy_bounds : `~astropy.units.Quantity`, optional
        Energy bounds used for the model overlay.
    ax : `matplotlib.axes.Axes`, optional
        Axes used for plotting.
    which : str, optional
        Sensitivity quantity to plot.
    which_label : str, optional
        Retained for API compatibility.
    int_sens_label : bool, optional
        Whether to append integrated sensitivity to labels.

    Returns
    -------
    `matplotlib.axes.Axes`
        Axes containing the plot.
    """
    del which_label

    if not isinstance(tables_south, list) or not isinstance(tables_north, list):
        raise TypeError("Both inputs must be lists.")

    ax = plt.gca() if ax is None else ax

    linestyle = [
        "solid",
        (0, (5, 1)),
        (0, (3, 5, 1, 5)),
    ]

    ax.set_prop_cycle(
        color=["blue", "blue", "blue"],
        linestyle=linestyle,
    )

    for table in tables_south:
        label = _get_irf_label(
            table,
            int_sens_label=int_sens_label,
        )
        plot_sensitivity_from_table(
            table,
            which=which,
            ax=ax,
            label=label,
        )

    ax.set_prop_cycle(
        color=["green", "green", "green"],
        linestyle=linestyle,
    )

    for table in tables_north:
        label = _get_irf_label(
            table,
            int_sens_label=int_sens_label,
        )
        plot_sensitivity_from_table(
            table,
            which=which,
            ax=ax,
            label=label,
        )

    if model is not None:
        plot_kwargs = {
            "ax": ax,
            "sed_type": "e2dnde",
        }

        model.spectral_model.plot(
            energy_bounds=energy_bounds,
            label=model.name,
            color="k",
            **plot_kwargs,
        )
        model.spectral_model.plot_error(
            energy_bounds=energy_bounds,
            **plot_kwargs,
        )

    ax.legend(
        loc="best",
        fontsize=7,
    )

    return ax


def plot_tables_sensitivity(
    tables_south,
    tables_north=None,
    model=None,
    sens_info=None,
    box_name=None,
    abs_model=None,
    which_label="both",
    file_path=None,
    **kwargs,
):
    """Create a complete sensitivity figure.

    Parameters
    ----------
    tables_south : list
        Sensitivity tables for the South site.
    tables_north : list, optional
        Sensitivity tables for the North site.
    model : object, optional
        Main model overlay.
    sens_info : str, optional
        Additional sensitivity information shown inside the axes.
    box_name : str, optional
        Label shown near the upper-left corner.
    abs_model : object, optional
        Optional absorbed model overlay.
    which_label : str, optional
        Retained for API compatibility.
    file_path : str or `~pathlib.Path`, optional
        Destination path used to save the figure.
    **kwargs : dict
        Optional ``energy_bounds``, ``ylim``, ``sed_type``,
        ``xaxis_label``, and ``yaxis_label`` entries.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Created figure.
    ax : `matplotlib.axes.Axes`
        Created axes.
    """
    if not isinstance(tables_south, list):
        raise TypeError("`tables_south` must be a list.")

    if tables_north and not isinstance(tables_north, list):
        raise TypeError("`tables_north` must be a list.")

    energy_bounds = kwargs.get(
        "energy_bounds",
        [3e-2, 1e2] * u.TeV,
    )
    ylim = kwargs.get(
        "ylim",
        [1e-14, 1e-8],
    )
    sed_type = kwargs.get(
        "sed_type",
        "e2dnde",
    )

    fig, ax = plt.subplots()

    if tables_north:
        plot_irfs_superpose(
            tables_south,
            tables_north,
            ax=ax,
            which_label=which_label,
            int_sens_label=False,
        )
    else:
        plot_irfs(
            tables_south,
            ax=ax,
            which_label=which_label,
            int_sens_label=False,
        )

    if model is not None:
        model.spectral_model.plot(
            label=model.name,
            energy_bounds=energy_bounds,
            ax=ax,
            sed_type=sed_type,
            color="black",
        )
        model.spectral_model.plot_error(
            energy_bounds=energy_bounds,
            ax=ax,
            sed_type=sed_type,
        )

        if abs_model is not None:
            abs_model.spectral_model.plot(
                label=abs_model.name,
                energy_bounds=energy_bounds,
                ax=ax,
                sed_type=sed_type,
                linestyle="--",
                color="black",
            )
            abs_model.spectral_model.plot_error(
                energy_bounds=energy_bounds,
                ax=ax,
                sed_type=sed_type,
            )

    ax.set_xlabel(
        kwargs.get(
            "xaxis_label",
            DEFAULT_XAXIS_LABEL["TeV"],
        )
    )
    ax.set_ylabel(
        kwargs.get(
            "yaxis_label",
            DEFAULT_YAXIS_LABEL[sed_type],
        )
    )
    ax.set_ylim(ylim)
    ax.set_xlim(energy_bounds.value)

    ax.tick_params(
        direction="in",
        which="both",
        top=True,
        right=True,
    )

    if box_name:
        ax.text(
            0.1,
            0.9,
            box_name,
            transform=ax.transAxes,
        )

    if sens_info:
        ax.text(
            0.1,
            0.05,
            sens_info,
            fontsize=6,
            transform=ax.transAxes,
        )

    handles, labels = ax.get_legend_handles_labels()

    if handles:
        ax.legend(frameon=False)

    if file_path:
        plt.savefig(file_path)

    return fig, ax
