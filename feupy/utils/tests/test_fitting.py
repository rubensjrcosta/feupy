# Licensed under a 3-clause BSD style license - see LICENSE

import astropy.units as u
from astropy.table import Table
from gammapy.modeling.models import PowerLawSpectralModel

from feupy.utils.fitting import fit_spectral_model_to_flux_points


def make_flux_points_table():
    """Create a flux-points table for testing."""
    table = Table(
        {
            "e_min": [1, 2, 3] * u.TeV,
            "e_max": [2, 3, 4] * u.TeV,
            "dnde": [1e-11, 2e-11, 1e-11] / (u.TeV * u.cm**2 * u.s),
            "dnde_err": [1e-12, 1e-12, 1e-12] / (u.TeV * u.cm**2 * u.s),
        }
    )
    table.meta["SED_TYPE"] = "dnde"

    return table


def test_fit_spectral_model_to_flux_points():
    table = make_flux_points_table()
    spectral_model = PowerLawSpectralModel()

    fitted_model = fit_spectral_model_to_flux_points(
        table,
        spectral_model,
    )

    assert isinstance(fitted_model, PowerLawSpectralModel)
    assert hasattr(fitted_model, "parameters")
    assert len(fitted_model.parameters.free_parameters.names) > 0


def test_fit_spectral_model_returns_input_model():
    table = make_flux_points_table()
    spectral_model = PowerLawSpectralModel()

    fitted_model = fit_spectral_model_to_flux_points(
        table,
        spectral_model,
    )

    assert fitted_model is spectral_model
