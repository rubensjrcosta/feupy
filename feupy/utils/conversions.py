# Licensed under a 3-clause BSD style license - see LICENSE
"""Physical unit conversion utilities."""

import astropy.units as u

__all__ = [
    "jy_to_erg_cm2_s",
    "frequency_to_energy",
]


def jy_to_erg_cm2_s(frequency, flux_density):
    """Convert flux density to energy flux.

    Parameters
    ----------
    frequency : `~astropy.units.Quantity`
        Frequency of the measurement.
    flux_density : `~astropy.units.Quantity`
        Spectral flux density, for example in Jy or mJy.

    Returns
    -------
    energy_flux : `~astropy.units.Quantity`
        Energy flux in ``erg cm-2 s-1``.
    """
    frequency = frequency.to(u.Hz)
    flux_density = flux_density.to(u.Jy)

    return (flux_density * frequency).to("erg cm-2 s-1")


def frequency_to_energy(frequency):
    """Convert frequency to photon energy.

    Parameters
    ----------
    frequency : `~astropy.units.Quantity`
        Frequency.

    Returns
    -------
    energy : `~astropy.units.Quantity`
        Photon energy in eV.
    """
    return frequency.to(u.eV, equivalencies=u.spectral())
