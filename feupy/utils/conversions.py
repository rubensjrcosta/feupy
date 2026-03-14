# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Physical unit conversions."""

import astropy.units as u

__all__ = [
    "jy_to_erg_cm2_s",
    "frequency_to_energy",
]


def jy_to_erg_cm2_s(freq, flux):
    """
    Convert flux density (Fν) to energy flux (νFν).

    Parameters
    ----------
    freq : `~astropy.units.Quantity`
        Frequency.
    flux : `~astropy.units.Quantity`
        Flux density (e.g. Jy or mJy).

    Returns
    -------
    flux : `~astropy.units.Quantity`
        Energy flux in ``erg cm-2 s-1``.
    """
    flux_density = flux.to(u.Jy)
    frequency = freq.to(u.Hz)

    return (flux_density * frequency).to("erg cm-2 s-1")


def frequency_to_energy(freq):
    """
    Convert frequency to photon energy.

    Parameters
    ----------
    freq : `~astropy.units.Quantity`
        Frequency.

    Returns
    -------
    energy : `~astropy.units.Quantity`
        Energy in electron volts.
    """
    return freq.to(u.eV, equivalencies=u.spectral())