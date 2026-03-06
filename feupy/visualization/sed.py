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

from feupy.visualization.styles.markers import make_marker_dict
from feupy.visualization import LINESTYLES_DEFAULT
from feupy.utils.datasets import get_energy_bounds_from_datasets
from feupy.visualization.utils.units import (
    DEFAULT_XAXIS_LABEL,
    DEFAULT_YAXIS_LABEL,
)

__all__ = ["SEDPlotter"]


class SEDPlotter:
    def __init__(self, datasets, models=None, sed_type="e2dnde"):
        self.datasets = datasets
        self.models = models
        self.sed_type = sed_type
        self.ax = None

    # -------------------------------
    # Helpers for Defaults
    # -------------------------------
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
            labelcolor="black",
            frameon=False,
        )

    # -------------------------------
    # Plot Components
    # -------------------------------
    def customize_legend(self, legend_kwargs):
        self.ax.legend(**legend_kwargs)

    def set_axis_labels(self, axis_kwargs):
        xlabel, ylabel = axis_kwargs["label"]
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    def set_axis_units(self, axis_kwargs):
        xunit, yunit = axis_kwargs["units"]
        self.ax.xaxis.set_units(u.Unit(xunit))
        self.ax.yaxis.set_units(u.Unit(yunit))

    def set_plot_limits(self, limits_kwargs):
        self.ax.set_xlim(limits_kwargs["energy_bounds"].value)
        self.ax.set_ylim(limits_kwargs["ylim"])

    # -------------------------------
    # Dataset Plotting
    # -------------------------------
    def plot_datasets(self, plot_kwargs, ref_markers):
        for dataset in self.datasets:
            kwargs_ds = {
                **ref_markers.get(dataset.name, {}),
                "ls": "None",
                "lw": 0.5,
                "markeredgecolor": "k",
                "mew": 0.4,
                "elinewidth": 0.6,
                "capsize": 1.5,
            }

            dataset.data.plot(**plot_kwargs, **kwargs_ds)

            # Model error band (dataset-level)
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

    # -------------------------------
    # Model Plotting
    # -------------------------------
    def plot_models(self, plot_kwargs, energy_bounds, show_error):
        if not self.models:
            return

        linestyle_cycle = itertools.cycle(LINESTYLES_DEFAULT)

        for model in self.models:
            spec = model.spectral_model

            kwargs_model = dict(
                label=model.name,
                linestyle=next(linestyle_cycle),
                color="black",
                marker=",",
                energy_bounds=energy_bounds,
            )

            spec.plot(**plot_kwargs, **kwargs_model)

            if show_error:
                spec.plot_error(
                    energy_bounds=energy_bounds,
                    alpha=0.05,
                    **plot_kwargs,
                )

    # -------------------------------
    # Main Plot Function
    # -------------------------------
    def plot(
        self,
        ax=None,
        file_path=None,
        ref_markers=None,
        box_name=None,
        error_band=False,
        **kwargs,
    ):
        # Axes
        self.ax = ax or plt.gca()

        # Defaults (user-overridable)
        axis_kwargs = kwargs.setdefault("axis", self._default_axis())
        limits_kwargs = kwargs.setdefault("limits", self._default_limits())
        legend_kwargs = kwargs.setdefault("kwargs_legend", self._default_legend())
        model_kwargs = kwargs.get("kwargs_models", {})

        # Units & labels
        self.set_axis_units(axis_kwargs)

        # Common plot kwargs
        plot_kwargs = dict(ax=self.ax, sed_type=self.sed_type)

        # Marker dictionary
        ref_names = list(self.datasets.names)
        if self.models:
            ref_names += list(self.models.names)

        if ref_markers is None: 
            ref_markers = make_marker_dict(
                labels=ref_names,
                marker="o",
                marker_size=4,
            )
    

        # Plot datasets
        self.plot_datasets(plot_kwargs, ref_markers)

        # Energy bounds for models
        energy_bounds = model_kwargs.get(
            "energy_bounds", limits_kwargs["energy_bounds"]
        )

        # Plot models
        self.plot_models(plot_kwargs, energy_bounds, error_band)

        # Axes formatting
        self.set_plot_limits(limits_kwargs)
        self.set_axis_labels(axis_kwargs)

        # Box label
        if box_name:
            self.ax.text(0.1, 0.9, box_name, transform=self.ax.transAxes)

        # Legend
        self.customize_legend(legend_kwargs)

        # Save
        if file_path:
            plt.savefig(file_path, bbox_inches="tight")

        return self.ax
