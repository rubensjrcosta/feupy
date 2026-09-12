# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spectral energy distribution plotting utilities."""

import itertools

import matplotlib.pyplot as plt
from astropy import units as u

from feupy.utils.datasets import get_energy_bounds_from_datasets
from feupy.visualization.styles.linestyles import LINESTYLES_DEFAULT
from feupy.visualization.styles.markers import build_fp_kwargs
from feupy.visualization.utils.labels import (
    DEFAULT_XAXIS_LABEL,
    DEFAULT_YAXIS_LABEL,
)

__all__ = ["SEDPlotter"]


class SEDPlotter:
    """Plot spectral energy distributions from datasets and models.

    Parameters
    ----------
    datasets : object
        Collection of datasets to plot. The object is expected to expose
        ``names`` and to be iterable.
    models : object, optional
        Collection of spectral models to plot.
    sed_type : str, optional
        SED representation passed to the plotting methods.
        Default is ``"e2dnde"``.
    """

    def __init__(self, datasets, models=None, sed_type="e2dnde"):
        self.datasets = datasets
        self.models = models
        self.sed_type = sed_type

    def _default_axis(self):
        return {
            "label": (
                DEFAULT_XAXIS_LABEL["TeV"],
                DEFAULT_YAXIS_LABEL[self.sed_type],
            ),
            "units": ("TeV", "TeV cm-2 s-1"),
        }

    def _default_limits(self):
        return {
            "energy_bounds": [1e-5, 2e3] * u.TeV,
            "ylim": [1e-23, 1e-7],
        }

    def _default_legend(self):
        return {
            "ncol": 3,
            "loc": "lower left",
            "markerscale": 0.75,
            "fontsize": 5,
            "frameon": False,
        }

    def _save_plot(self, file_path):
        """Save the current figure.

        Parameters
        ----------
        file_path : str or `~pathlib.Path`
            Destination path. If None, the figure is not saved.
        """
        if file_path:
            plt.savefig(file_path, dpi=300, bbox_inches="tight")

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
                spectral_model = dataset.models[dataset.name].spectral_model
                spectral_model.plot_error(
                    **plot_kwargs,
                    energy_bounds=energy_bounds,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.2,
                )

    def _plot_models(
        self,
        ax,
        plot_kwargs,
        energy_bounds,
        show_error,
        ref_markers,
    ):
        if not self.models:
            return

        linestyle_cycle = itertools.cycle(LINESTYLES_DEFAULT)

        for model in self.models:
            spectral_model = model.spectral_model
            color = ref_markers.get(model.name, {}).get("color", "black")

            kwargs_model = {
                "label": model.name,
                "linestyle": next(linestyle_cycle),
                "color": color,
                "marker": ",",
                "energy_bounds": energy_bounds,
            }

            spectral_model.plot(**plot_kwargs, **kwargs_model)

            if show_error:
                spectral_model.plot_error(
                    energy_bounds=energy_bounds,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.05,
                    **plot_kwargs,
                )

    def plot(
        self,
        ax=None,
        file_path=None,
        ref_markers=None,
        box_name=None,
        error_band=False,
        **kwargs,
    ):
        """Plot datasets and spectral models.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`, optional
            Axes used for plotting. If None, the current axes are used.
        file_path : str or `~pathlib.Path`, optional
            Destination path used to save the figure.
        ref_markers : dict, optional
            Plotting keyword arguments indexed by dataset or model name.
        box_name : str, optional
            Text displayed inside the axes.
        error_band : bool, optional
            Whether to draw model uncertainty bands. Default is False.
        **kwargs : dict
            Optional ``axis``, ``limits``, ``kwargs_legend``, and
            ``kwargs_models`` dictionaries.

        Returns
        -------
        `matplotlib.axes.Axes`
            Axes containing the SED plot.
        """
        ax = ax or plt.gca()

        axis_kwargs = kwargs.get("axis", self._default_axis())
        limits_kwargs = kwargs.get("limits", self._default_limits())
        legend_kwargs = kwargs.get("kwargs_legend", self._default_legend())
        model_kwargs = kwargs.get("kwargs_models", {})

        self._set_axis_units(ax, axis_kwargs)

        plot_kwargs = {
            "ax": ax,
            "sed_type": self.sed_type,
        }

        ref_names = list(self.datasets.names)

        if self.models:
            ref_names += list(self.models.names)

        if ref_markers is None:
            ref_markers = build_fp_kwargs(
                labels=ref_names,
                marker_size=4,
            )

        self._plot_datasets(ax, plot_kwargs, ref_markers)

        energy_bounds = model_kwargs.get(
            "energy_bounds",
            limits_kwargs["energy_bounds"],
        )

        self._plot_models(
            ax,
            plot_kwargs,
            energy_bounds,
            error_band,
            ref_markers,
        )

        self._set_plot_limits(ax, limits_kwargs)
        self._set_axis_labels(ax, axis_kwargs)

        if box_name:
            ax.text(
                0.1,
                0.9,
                box_name,
                fontsize=8,
                transform=ax.transAxes,
            )

        ax.legend(**legend_kwargs)
        self._save_plot(file_path)

        return ax
