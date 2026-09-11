# Licensed under a 3-clause BSD style license - see LICENSE
"""Fit and SED plotting utilities."""

import matplotlib.pyplot as plt
from gammapy.modeling import Fit
from gammapy.modeling.models import Models

from feupy.utils.aic import calculate_aic
from feupy.visualization.sed import SEDPlotter
from feupy.visualization.styles.markers.plotting import build_fp_kwargs

__all__ = ["run_fit_and_plot"]


def run_fit_and_plot(
    datasets,
    model,
    fitter=None,
    show_plot=True,
    show_result_fit=True,
    **kwargs,
):
    """Fit a model to datasets and optionally plot the resulting SED.

    Parameters
    ----------
    datasets : `~gammapy.datasets.Datasets`
        Datasets used in the fit.
    model : `~gammapy.modeling.models.SkyModel`
        Model assigned to the datasets.
    fitter : `~gammapy.modeling.Fit`, optional
        Fitter instance. If not provided, a default `~gammapy.modeling.Fit`
        instance is created.
    show_plot : bool, optional
        Whether to create and display the SED plot. Default is True.
    show_result_fit : bool, optional
        Whether to print the fit result and AIC value. Default is True.
    **kwargs
        Additional keyword arguments passed to
        `~feupy.visualization.sed.SEDPlotter.plot`.

    Returns
    -------
    result_fit : `~gammapy.modeling.FitResult`
        Fit result.
    """
    datasets_in = datasets.copy()
    datasets.models = Models([model])

    fitter = fitter or Fit()
    result_fit = fitter.run(datasets=datasets)

    if show_result_fit:
        print(result_fit)
        print(f"AIC: {calculate_aic(datasets, result_fit)}")

    if show_plot:
        ref_markers = build_fp_kwargs(datasets_in.names, marker_size=4)

        sed_plotter = SEDPlotter(
            datasets=datasets_in,
            models=Models([model.copy(name=f"FIT {model.spectral_model.tag[1]}")]),
        )

        plot_kwargs = {"ref_markers": ref_markers}
        plot_kwargs.update(kwargs)

        sed_plotter.plot(**plot_kwargs)
        plt.show()

    return result_fit
