# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from astropy.table import Table
from astropy import units as u

from gammapy.modeling.models import PowerLawSpectralModel

from feupy.utils.fitting import fit_spectral_model_to_flux_points


# ---------------------------------------------------------------------
# Helper: fake flux points table (Gammapy-compliant)
# ---------------------------------------------------------------------
def make_fake_flux_points():
    table = Table(
        {
            "e_min": [1, 2, 3] * u.TeV,
            "e_max": [2, 3, 4] * u.TeV,
            "dnde": [1e-11, 2e-11, 1e-11]
            * (1 / (u.TeV * u.cm**2 * u.s)),
            "dnde_err": [1e-12, 1e-12, 1e-12]
            * (1 / (u.TeV * u.cm**2 * u.s)),
        }
    )

    # REQUIRED by Gammapy
    table.meta["SED_TYPE"] = "dnde"

    return table


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_fit_spectral_model_basic():
    table = make_fake_flux_points()
    model = PowerLawSpectralModel()

    fitted_model = fit_spectral_model_to_flux_points(table, model)

    assert fitted_model is not None


def test_fit_output_type():
    table = make_fake_flux_points()
    model = PowerLawSpectralModel()

    fitted_model = fit_spectral_model_to_flux_points(table, model)

    assert hasattr(fitted_model, "parameters")
    assert len(fitted_model.parameters.free_parameters.names) > 0


def test_fit_does_not_raise():
    table = make_fake_flux_points()
    model = PowerLawSpectralModel()

    try:
        fit_spectral_model_to_flux_points(table, model)
    except Exception as e:
        pytest.fail(f"Fit raised unexpected exception: {e}")

