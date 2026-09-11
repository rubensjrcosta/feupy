# Licensed under a 3-clause BSD style license - see LICENSE

import astropy.units as u
import pytest

from feupy.utils.conversions import (
    frequency_to_energy,
    jy_to_erg_cm2_s,
)


@pytest.mark.parametrize(
    ("frequency", "flux_density"),
    [
        (1e14 * u.Hz, 1 * u.Jy),
        (1e14 * u.Hz, 1 * u.mJy),
    ],
)
def test_jy_to_erg_cm2_s(frequency, flux_density):
    result = jy_to_erg_cm2_s(frequency, flux_density)
    expected = (flux_density.to(u.Jy) * frequency.to(u.Hz)).to("erg cm-2 s-1")

    assert result.unit == u.Unit("erg cm-2 s-1")
    assert result.value == pytest.approx(expected.value)


@pytest.mark.parametrize(
    "frequency",
    [
        1e14 * u.Hz,
        100 * u.GHz,
    ],
)
def test_frequency_to_energy(frequency):
    result = frequency_to_energy(frequency)
    expected = frequency.to(u.eV, equivalencies=u.spectral())

    assert result.unit == u.eV
    assert result.value == pytest.approx(expected.value)
