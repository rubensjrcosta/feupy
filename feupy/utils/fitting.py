# Licensed under a 3-clause BSD style license - see LICENSE
"""Spectral fitting utilities."""

from gammapy.datasets import Datasets, FluxPointsDataset
from gammapy.estimators import FluxPoints
from gammapy.modeling import Fit
from gammapy.modeling.models import Models, SkyModel

__all__ = ["fit_spectral_model_to_flux_points"]


def fit_spectral_model_to_flux_points(flux_points_table, spectral_model):
    """Fit a spectral model to flux points.

    Parameters
    ----------
    flux_points_table : `~astropy.table.Table`
        Table containing the flux-point measurements.
    spectral_model : `~gammapy.modeling.models.SpectralModel`
        Spectral model to fit.

    Returns
    -------
    spectral_model : `~gammapy.modeling.models.SpectralModel`
        Best-fit spectral model.
    """
    flux_points = FluxPoints.from_table(flux_points_table)
    dataset = FluxPointsDataset(data=flux_points)
    datasets = Datasets([dataset])

    model = SkyModel(spectral_model=spectral_model)
    datasets.models = Models([model])

    Fit().run(datasets=datasets)

    return model.spectral_model
