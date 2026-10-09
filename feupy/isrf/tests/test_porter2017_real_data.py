# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Regression tests using the Porter et al. (2017) GALPROP ISRF data."""

import os

import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord

from feupy.isrf import Porter2017ISRF
from feupy.isrf.utils import blackbody_sed, integrate_isrf_sed


pytestmark = pytest.mark.skipif(
    not os.environ.get("FEUPY_DATA"),
    reason="FEUPY_DATA is not configured.",
)

SED_UNIT = u.eV / u.cm**3
TEMPERATURES = {
    "FIR": 40 * u.K,
    "NIR": 500 * u.K,
    "VIS": 3500 * u.K,
    "UV": 20000 * u.K,
}
REFERENCE_DENSITIES = {
    "FIR": 0.8288,
    "NIR": 0.3019,
    "VIS": 1.4914,
    "UV": 1.2304,
}


@pytest.fixture(scope="module")
def isrf():
    """Load the R12 GALPROP ISRF model."""
    return Porter2017ISRF(model="R12")


@pytest.fixture(scope="module")
def position():
    """Reference Galactic position."""
    return SkyCoord(
        l=17.8 * u.deg,
        b=-0.7 * u.deg,
        distance=4.0 * u.kpc,
        frame="galactic",
    )


@pytest.fixture(scope="module")
def fit_4bb(isrf, position):
    """Compute the validated four-blackbody NNLS approximation once."""
    return isrf.fit_blackbody_components(
        position,
        temperatures=TEMPERATURES,
        method="nnls",
        threshold=0.01,
    )


def test_reference_energy_density(isrf, position):
    """Check the reference integrated energy density."""
    energy_density = isrf.energy_density(position)
    np.testing.assert_allclose(
        energy_density.to_value(SED_UNIT),
        3.909067053838413,
        rtol=1e-6,
    )


def test_component_energy_density_sum(isrf, position):
    """Check conservation of total energy density across FITS components."""
    components = ("Direct", "Scattered", "Transient", "Thermal")
    total = isrf.energy_density(position, component="Total")
    component_sum = sum(
        (isrf.energy_density(position, component=name) for name in components),
        0 * SED_UNIT,
    )
    np.testing.assert_allclose(
        total.to_value(SED_UNIT),
        component_sum.to_value(SED_UNIT),
        rtol=1e-12,
    )


def test_reference_energy_grid(isrf):
    """Check the monotonically increasing photon-energy grid."""
    energy = isrf.energy
    assert energy.size == 128
    assert np.all(np.diff(energy.to_value(u.eV)) > 0)


@pytest.mark.parametrize("name,temperature", list(TEMPERATURES.items()))
def test_unit_blackbody_normalization(isrf, name, temperature):
    """A unit blackbody integrates to approximately 1 eV/cm3 on the grid."""
    sed = blackbody_sed(
        isrf.energy,
        temperature=temperature,
        energy_density=1 * SED_UNIT,
    )
    density = integrate_isrf_sed(isrf.energy, sed)
    np.testing.assert_allclose(density.to_value(SED_UNIT), 1.0, rtol=0.05)


def test_nnls_reference_fit(fit_4bb):
    """Reproduce the validated NNLS normalizations at the reference point."""
    assert fit_4bb["success"]
    assert fit_4bb["method"] == "nnls"
    assert list(fit_4bb["energy_densities"]) == list(TEMPERATURES)
    for name, expected in REFERENCE_DENSITIES.items():
        np.testing.assert_allclose(
            fit_4bb["energy_densities"][name].to_value(SED_UNIT),
            expected,
            rtol=2e-3,
        )
        assert fit_4bb["energy_densities"][name] >= 0 * SED_UNIT


def test_nnls_fit_quality(fit_4bb):
    """Check the spectral and integrated-energy fit metrics."""
    np.testing.assert_allclose(
        fit_4bb["relative_integrated_difference"], -0.0280, atol=5e-4
    )
    np.testing.assert_allclose(
        fit_4bb["rms_log_residual"], 0.22218720242201664, atol=2e-3
    )
    np.testing.assert_allclose(
        fit_4bb["reference_integrated_density"].to_value(SED_UNIT),
        3.909067053838413,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        fit_4bb["fitted_integrated_density"].to_value(SED_UNIT),
        3.7997,
        rtol=5e-4,
    )


def test_nnls_component_spectra_sum(fit_4bb):
    """Check that the component spectra sum to the fitted spectrum."""
    summed = sum(fit_4bb["components"].values(), 0 * SED_UNIT)
    np.testing.assert_allclose(
        summed.to_value(SED_UNIT),
        fit_4bb["fit_sed"].to_value(SED_UNIT),
        rtol=1e-12,
        atol=1e-30,
    )
    np.testing.assert_allclose(
        fit_4bb["bolometric_density"].to_value(SED_UNIT),
        sum(value.to_value(SED_UNIT) for value in fit_4bb["energy_densities"].values()),
        rtol=1e-12,
    )


def test_nnls_default_temperatures(isrf, position, fit_4bb):
    """Check that the default configuration matches the validated 4BB fit."""
    default_fit = isrf.fit_blackbody_components(position)
    assert default_fit["method"] == "nnls"
    for name in TEMPERATURES:
        assert default_fit["temperatures"][name] == TEMPERATURES[name]
        np.testing.assert_allclose(
            default_fit["energy_densities"][name].to_value(SED_UNIT),
            fit_4bb["energy_densities"][name].to_value(SED_UNIT),
            rtol=1e-12,
        )


def test_naima_blackbody_fields(isrf, position):
    """Check the Naima thermal-field format and optional CMB."""
    fields = isrf.to_naima_blackbodies(position, include_cmb=True)
    assert len(fields) == 5
    assert fields[-1] == "CMB"
    for field, (name, temperature) in zip(fields[:-1], TEMPERATURES.items()):
        assert len(field) == 3
        assert field[0] == name
        assert field[1] == temperature
        np.testing.assert_allclose(
            field[2].to_value(SED_UNIT), REFERENCE_DENSITIES[name], rtol=2e-3
        )


def test_naima_blackbody_fields_without_cmb(isrf, position):
    """Do not add the CMB unless explicitly requested."""
    fields = isrf.to_naima_blackbodies(position)
    assert len(fields) == 4
    assert all(field[0] != "CMB" for field in fields)


@pytest.mark.parametrize("method", ["nnls", "linear", "log", "relative"])
def test_fit_methods_return_finite_nonnegative_densities(isrf, position, method):
    """All supported objectives should return physically admissible fields."""
    fit = isrf.fit_blackbody_components(position, method=method)
    assert fit["success"]
    assert np.isfinite(fit["rms_log_residual"])
    assert np.isfinite(fit["relative_integrated_difference"])
    assert all(
        np.isfinite(value.to_value(SED_UNIT)) and value >= 0 * SED_UNIT
        for value in fit["energy_densities"].values()
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"method": "invalid"}, "method"),
        ({"threshold": -0.1}, "threshold"),
        ({"threshold": 1.0}, "threshold"),
        ({"temperatures": {}}, "temperatures"),
        ({"energy_range": (1 * u.eV, 0.1 * u.eV)}, "energy_range"),
    ],
)
def test_invalid_fit_configuration(isrf, position, kwargs, match):
    """Reject invalid fitting configurations with informative errors."""
    with pytest.raises(ValueError, match=match):
        isrf.fit_blackbody_components(position, **kwargs)


def test_fit_rejects_vector_positions(isrf, position):
    """The blackbody fit currently requires a scalar sky position."""
    positions = SkyCoord([position, position])
    with pytest.raises(ValueError, match="scalar"):
        isrf.fit_blackbody_components(positions)
