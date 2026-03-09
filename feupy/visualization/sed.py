# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
SEDPlotter class.

This module provides the SEDPlotter class, which is used to plot Spectral Energy Distributions (SEDs)
from a collection of datasets and models. It offers flexible options for customizing plot appearance,
legends, axis labels, units, and plot limits.
"""

import itertools
import matplotlib.pyplot as plt
from astropy import units as u

from feupy.visualization.styles.markers import build_fp_kwargs
from feupy.visualization.styles.linestyles import LINESTYLES_DEFAULT
from feupy.utils.datasets import get_energy_bounds_from_datasets
from feupy.visualization.utils.units import (
    DEFAULT_XAXIS_LABEL,
    DEFAULT_YAXIS_LABEL,
)


class SEDPlotter:
    """
    Pipeline-safe SED plot renderer.
    """

    def __init__(self, datasets, models=None, sed_type="e2dnde"):
        self.datasets = datasets
        self.models = models
        self.sed_type = sed_type

    # -------------------------------------------------
    # Defaults
    # -------------------------------------------------

    def _default_axis(self):
        return dict(
            label=(DEFAULT_XAXIS_LABEL["TeV"], DEFAULT_YAXIS_LABEL[self.sed_type]),
            units=("TeV", "TeV cm-2 s-1"),
        )

    def _default_limits(self):
        return dict(
            energy_bounds=[1e-5, 2e3] * u.TeV,
            ylim=[1e-23, 1e-7],
        )

    def _default_legend(self):
        return dict(
            ncol=3,
            loc="lower left",
            markerscale=0.75,
            fontsize=5,
            frameon=False,
        )

    # -------------------------------------------------
    # Axis formatting
    # -------------------------------------------------

    def _save_plot(self, file_path):
        """
        Save the SED to a file.

        Parameters
        ----------
        file_path : str or `~pathlib.Path`
            File path or name where the plot will be saved.
        """
        if file_path:
            plt.savefig(file_path, bbox_inches='tight')
            
    def _set_axis_labels(self, ax, axis_kwargs):
        xlabel, ylabel = axis_kwargs["label"]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def _set_axis_units(self, ax, axis_kwargs):
        xunit, yunit = axis_kwargs["units"]
        ax.xaxis.set_units(u.Unit(xunit))
        ax.yaxis.set_units(u.Unit(yunit))

    def _set_plot_limits(self, ax, limits_kwargs):
        ax.set_xlim(limits_kwargs["energy_bounds"].value)
        ax.set_ylim(limits_kwargs["ylim"])

    # -------------------------------------------------
    # Dataset rendering
    # -------------------------------------------------

    def _plot_datasets(self, ax, plot_kwargs, ref_markers):
        for dataset in self.datasets:

            kwargs_ds = {
                **ref_markers.get(dataset.name, {}),
                "ls": "None",
                "lw": 0.5,
                "markeredgecolor": "black",
                "mew": 0.4,
                "elinewidth": 0.6,
                "capsize": 1.5,
                "zorder": 3,
            }

            dataset.data.plot(**plot_kwargs, **kwargs_ds)

            color = kwargs_ds.get("color", "black")

            energy_bounds = get_energy_bounds_from_datasets(dataset)

            if dataset.models and dataset.name in dataset.models.names:
                spec = dataset.models[dataset.name].spectral_model

                spec.plot_error(
                    **plot_kwargs,
                    energy_bounds=energy_bounds,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.2,
                )

    # -------------------------------------------------
    # Model rendering
    # -------------------------------------------------

    def _plot_models(self, ax, plot_kwargs, energy_bounds, show_error, ref_markers):
        if not self.models:
            return

        linestyle_cycle = itertools.cycle(LINESTYLES_DEFAULT)

        for model in self.models:

            spec = model.spectral_model

            color = ref_markers.get(model.name, {}).get("color", "black")
            
            kwargs_model = dict(
                label=model.name,
                linestyle=next(linestyle_cycle),
                color=color,
                marker=",",
                energy_bounds=energy_bounds,
            )

            spec.plot(**plot_kwargs, **kwargs_model)

            if show_error:
                spec.plot_error(
                    energy_bounds=energy_bounds,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.05,
                    **plot_kwargs,
                )
    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def plot(
        self,
        ax=None,
        file_path=None,
        ref_markers=None,
        box_name=None,
        error_band=False,
        **kwargs,
    ):
        ax = ax or plt.gca()

        axis_kwargs = kwargs.get("axis", self._default_axis())
        limits_kwargs = kwargs.get("limits", self._default_limits())
        legend_kwargs = kwargs.get("kwargs_legend", self._default_legend())
        model_kwargs = kwargs.get("kwargs_models", {})

        self._set_axis_units(ax, axis_kwargs)

        plot_kwargs = dict(ax=ax, sed_type=self.sed_type)

        # Marker registry
        ref_names = list(self.datasets.names)

        if self.models:
            ref_names += list(self.models.names)

        if ref_markers is None:
            ref_markers = build_fp_kwargs(
                labels=ref_names,
                marker="o",
                marker_size=4,
            )

        # Plot components
        self._plot_datasets(ax, plot_kwargs, ref_markers)

        energy_bounds = model_kwargs.get(
            "energy_bounds",
            limits_kwargs["energy_bounds"],
        )

        self._plot_models(ax, plot_kwargs, energy_bounds, error_band, ref_markers)

        self._set_plot_limits(ax, limits_kwargs)
        self._set_axis_labels(ax, axis_kwargs)

        if box_name:
            ax.text(
                0.1,
                0.9,
                box_name,
                transform=ax.transAxes,
            )

        ax.legend(**legend_kwargs)
        
        # Save plot if file_path is provided
        self._save_plot(file_path)
        
        return ax