# Licensed under a 3-clause BSD style license
"""Sensitivity plotting utilities."""

import matplotlib.pyplot as plt
import astropy.units as u

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


# =====================
# Core plot
# =====================
def plot_sensitivity_from_table(
    sens_table, which="e2dnde", ax=None, plot_xerr=False, **kwargs
):
    """Plot sensitivity from table."""

    ax = plt.gca() if ax is None else ax

    e = sens_table["e_ref"]
    s = sens_table[which]

    xlabel = f"Energy [{e.unit.to_string(UNIT_STRING_FORMAT)}]"
    ylabel = {
        "excess": "Excess counts",
        "background": "Background counts",
        "on_radii": f"On region radius [{s.unit.to_string(UNIT_STRING_FORMAT)}]",
        "e2dnde": f"Flux Sensitivity [{s.unit.to_string(UNIT_STRING_FORMAT)}]",
    }.get(which, f"{which} [{s.unit.to_string(UNIT_STRING_FORMAT)}]")

    xerr = None
    if plot_xerr and "e_min" in sens_table.colnames:
        xerr = (sens_table["e_max"] - sens_table["e_min"]) / 2

    with quantity_support():
        ax.errorbar(e, s, xerr=xerr, **kwargs)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


# =====================
# IRF plotting (single site)
# =====================
def plot_irfs(
    tables,
    model=None,
    energy_bounds=None,
    ax=None,
    which="e2dnde",
    which_label="both",  # mantido por compatibilidade
    int_sens_label=True,
):

    if not isinstance(tables, list):
        raise TypeError("`tables` must be a list of tables.")

    ax = plt.gca() if ax is None else ax

    linestyle = ["solid", "solid", "solid"]
    ax.set_prop_cycle(color=["blue", "red", "green"], linestyle=linestyle)

    for table in tables:

        # ✅ NOVO: usa direto meta (sem Irfs)
        label = table.meta.get("IRF_NAME", "IRF")

        # integrated sensitivity
        if int_sens_label and "INT_SENS" in table.meta:
            int_sens = u.Quantity(table.meta["INT_SENS"])
            unit = int_sens.unit.to_string(UNIT_STRING_FORMAT)
            label += f" ({int_sens.value:.2e} {unit})"

        plot_sensitivity_from_table(
            table,
            which=which,
            ax=ax,
            label=label,
        )

    # model overlay
    if model is not None:
        kwargs = {"ax": ax, "sed_type": "e2dnde"}
        model.spectral_model.plot(
            energy_bounds=energy_bounds,
            label=model.name,
            color="k",
            **kwargs,
        )
        model.spectral_model.plot_error(
            energy_bounds=energy_bounds,
            **kwargs,
        )

    ax.legend(loc="best", fontsize=7)

    return ax


# =====================
# IRF plotting (South + North)
# =====================
def plot_irfs_superpose(
    tables_south,
    tables_north,
    model=None,
    energy_bounds=None,
    ax=None,
    which="e2dnde",
    which_label="both",  # mantido
    int_sens_label=True,
):

    if not isinstance(tables_south, list) or not isinstance(tables_north, list):
        raise TypeError("Both inputs must be lists.")

    ax = plt.gca() if ax is None else ax

    linestyle = ["solid", (0, (5, 1)), (0, (3, 5, 1, 5))]

    # SOUTH
    ax.set_prop_cycle(color=["blue", "blue", "blue"], linestyle=linestyle)

    for table in tables_south:

        label = table.meta.get("IRF_NAME", "IRF")

        if int_sens_label and "INT_SENS" in table.meta:
            int_sens = u.Quantity(table.meta["INT_SENS"])
            unit = int_sens.unit.to_string(UNIT_STRING_FORMAT)
            label += f" ({int_sens.value:.2e} {unit})"

        plot_sensitivity_from_table(table, which=which, ax=ax, label=label)

    # NORTH
    ax.set_prop_cycle(color=["green", "green", "green"], linestyle=linestyle)

    for table in tables_north:

        label = table.meta.get("IRF_NAME", "IRF")

        if int_sens_label and "INT_SENS" in table.meta:
            int_sens = u.Quantity(table.meta["INT_SENS"])
            unit = int_sens.unit.to_string(UNIT_STRING_FORMAT)
            label += f" ({int_sens.value:.2e} {unit})"

        plot_sensitivity_from_table(table, which=which, ax=ax, label=label)

    # model
    if model is not None:
        kwargs = {"ax": ax, "sed_type": "e2dnde"}
        model.spectral_model.plot(
            energy_bounds=energy_bounds,
            label=model.name,
            color="k",
            **kwargs,
        )
        model.spectral_model.plot_error(energy_bounds=energy_bounds, **kwargs)

    ax.legend(loc="best", fontsize=7)

    return ax


# =====================
# High-level plot
# =====================
def plot_tables_sensitivity(
    tables_south,
    tables_north=None,
    model=None,
    sens_info=None,
    box_name=None,
    abs_model=None,
    which_label="both",  # mantido
    file_path=None,
    **kwargs,
):

    if not isinstance(tables_south, list):
        raise TypeError("`tables_south` must be a list.")

    if tables_north and not isinstance(tables_north, list):
        raise TypeError("`tables_north` must be a list.")

    energy_bounds = kwargs.get("energy_bounds", [3e-2, 1e2] * u.TeV)
    ylim = kwargs.get("ylim", [1e-14, 1e-8])
    sed_type = kwargs.get("sed_type", "e2dnde")

    fig, ax = plt.subplots()

    # choose mode
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

    # model
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

    ax.set_xlabel(kwargs.get("xaxis_label", DEFAULT_XAXIS_LABEL["TeV"]))
    ax.set_ylabel(kwargs.get("yaxis_label", DEFAULT_YAXIS_LABEL[sed_type]))
    ax.set_ylim(ylim)
    ax.set_xlim(energy_bounds.value)

    ax.tick_params(direction="in", which="both", top=True, right=True)

    if box_name:
        ax.text(0.1, 0.9, box_name, transform=ax.transAxes)

    if sens_info:
        ax.text(0.1, 0.05, sens_info, fontsize=6, transform=ax.transAxes)

    ax.legend(frameon=False)

    if file_path:
        plt.savefig(file_path)

    return fig, ax