# Licensed under a 3-clause BSD style license - see LICENSE.rst

import astropy.units as u

from feupy.utils.conversions import (
    jy_to_erg_cm2_s,
    frequency_to_energy,
)


def test_jy_to_erg_cm2_s():
    freq = 1e14 * u.Hz
    flux = 1 * u.Jy

    result = jy_to_erg_cm2_s(freq, flux)

    expected = (flux * freq).to("erg cm-2 s-1")

    assert result.unit == u.Unit("erg cm-2 s-1")
    assert result.value == expected.value


def test_jy_to_erg_cm2_s_mjy():
    freq = 1e14 * u.Hz
    flux = 1 * u.mJy

    result = jy_to_erg_cm2_s(freq, flux)

    expected = (flux.to(u.Jy) * freq).to("erg cm-2 s-1")

    assert result.unit == u.Unit("erg cm-2 s-1")
    assert result.value == expected.value


def test_frequency_to_energy():
    freq = 1e14 * u.Hz

    energy = frequency_to_energy(freq)

    expected = freq.to(u.eV, equivalencies=u.spectral())

    assert energy.unit == u.eV
    assert energy.value == expected.value


def test_frequency_to_energy_ghz():
    freq = 100 * u.GHz

    energy = frequency_to_energy(freq)

    expected = freq.to(u.eV, equivalencies=u.spectral())

    assert energy.unit == u.eV
    assert energy.value == expected.value