# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Spectral fitting utilities.
"""

from gammapy.modeling import Fit
from gammapy.datasets import Datasets, FluxPointsDataset
from gammapy.estimators import FluxPoints
from gammapy.modeling.models import SkyModel, Models

__all__ = ["fit_spectral_model_to_flux_points"]


def fit_spectral_model_to_flux_points(flux_points_table, spectral_model):
    """Fit spectral model to flux points."""

    flux_points = FluxPoints.from_table(flux_points_table)
    datasets = Datasets(FluxPointsDataset(data=flux_points))

    model = SkyModel(spectral_model=spectral_model)
    datasets.models = Models([model])

    Fit().run(datasets=datasets)

    return model.spectral_model