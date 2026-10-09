# Licensed under a 3-clause BSD style license - see LICENSE
"""Tests for physical unit conversion utilities."""

import astropy.units as u
import pytest

from astropy.constants import c, h
from astropy.tests.helper import assert_quantity_allclose

from feupy.utils.conversions import (
    frequency_to_energy,
    flux_density_to_nu_fnu,
    wavelength_to_energy,
)


@pytest.mark.parametrize(
    ("frequency", "flux_density"),
    [
        (1e14 * u.Hz, 1 * u.Jy),
        (1e14 * u.Hz, 1 * u.mJy),
        (1 * u.GHz, 1000 * u.mJy),
    ],
)
def test_flux_density_to_nu_fnu(frequency, flux_density):
    """Test spectral flux density to energy flux conversion."""
    result = flux_density_to_nu_fnu(frequency, flux_density)

    expected = (
        frequency.to(u.Hz)
        * flux_density.to(u.Jy)
    ).to(u.erg / (u.cm**2 * u.s))

    assert_quantity_allclose(result, expected)


@pytest.mark.parametrize(
    "frequency",
    [
        1e14 * u.Hz,
        100 * u.GHz,
        1 * u.THz,
    ],
)
def test_frequency_to_energy(frequency):
    """Test frequency to photon energy conversion."""
    result = frequency_to_energy(frequency)

    expected = (h * frequency).to(u.eV)

    assert_quantity_allclose(result, expected)


@pytest.mark.parametrize(
    "wavelength",
    [
        500 * u.nm,
        1 * u.um,
        10 * u.cm,
    ],
)
def test_wavelength_to_energy(wavelength):
    """Test wavelength to photon energy conversion."""
    result = wavelength_to_energy(wavelength)

    expected = (h * c / wavelength).to(u.eV)

    assert_quantity_allclose(result, expected)


@pytest.mark.parametrize(
    "frequency",
    [
        1 * u.kg,
        1 * u.K,
    ],
)
def test_frequency_to_energy_invalid_units(frequency):
    """Test rejection of incompatible frequency units."""
    with pytest.raises(u.UnitConversionError):
        frequency_to_energy(frequency)


@pytest.mark.parametrize(
    "wavelength",
    [
        1 * u.kg,
        1 * u.K,
    ],
)
def test_wavelength_to_energy_invalid_units(wavelength):
    """Test rejection of incompatible wavelength units."""
    with pytest.raises(u.UnitConversionError):
        wavelength_to_energy(wavelength)


@pytest.mark.parametrize(
    ("frequency", "flux_density"),
    [
        (1 * u.kg, 1 * u.Jy),
        (1 * u.GHz, 1 * u.kg),
    ],
)
def test_flux_density_to_nu_fnu_invalid_units(frequency, flux_density):
    """Test rejection of incompatible flux conversion units."""
    with pytest.raises(u.UnitConversionError):
        flux_density_to_nu_fnu(frequency, flux_density)