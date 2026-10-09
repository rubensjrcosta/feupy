"""Tests for ISRF spectral utilities."""

import numpy as np
import pytest
import astropy.units as u

from feupy.isrf.utils import (
    add_cmb_to_isrf,
    blackbody_sed,
    cmb_energy_density,
    integrate_isrf_sed,
    sed_to_photon_density,
)


def test_blackbody_normalization():
    energy = np.geomspace(1e-6, 100, 10000) * u.eV
    sed = blackbody_sed(energy, 40 * u.K, 1 * u.eV / u.cm**3)
    actual = integrate_isrf_sed(energy, sed)
    assert np.isclose(actual.to_value(u.eV / u.cm**3), 1, rtol=1e-7)


def test_photon_density_roundtrip():
    energy = np.geomspace(1e-3, 10, 100) * u.eV
    sed = np.linspace(0.1, 2, 100) * u.eV / u.cm**3
    density = sed_to_photon_density(energy, sed)
    assert density.unit.is_equivalent(1 / (u.eV * u.cm**3))
    assert u.allclose(energy**2 * density, sed, rtol=1e-12)


def test_batched_sed_integration():
    energy = np.geomspace(1, 100, 101) * u.eV
    sed = np.stack([np.ones(101), 2 * np.ones(101)]) * u.eV / u.cm**3
    result = integrate_isrf_sed(energy, sed)
    expected = np.array([1, 2]) * np.log(100)
    np.testing.assert_allclose(result.to_value(u.eV / u.cm**3), expected, rtol=1e-12)


def test_cmb_energy_density():
    assert np.isclose(
        cmb_energy_density().to_value(u.eV / u.cm**3),
        0.2605705782580934,
        rtol=1e-6,
    )


def test_cmb_addition():
    energy = np.geomspace(1e-6, 10, 500) * u.eV
    zero = np.zeros(500) * u.eV / u.cm**3
    actual = add_cmb_to_isrf(energy, zero)
    expected = blackbody_sed(energy, 2.7255 * u.K, cmb_energy_density())
    assert u.allclose(actual, expected, rtol=1e-12)


@pytest.mark.parametrize("bad_energy", [
    np.array([1, 1, 2]),
    np.array([0, 1, 2]),
    np.array([2, 1, 3]),
])
def test_invalid_energy(bad_energy):
    with pytest.raises(ValueError):
        integrate_isrf_sed(
            bad_energy * u.eV,
            np.ones(3) * u.eV / u.cm**3,
        )


def test_invalid_sed():
    with pytest.raises(ValueError):
        sed_to_photon_density(
            np.array([1, 2, 3]) * u.eV,
            np.array([1, -1, 2]) * u.eV / u.cm**3,
        )
