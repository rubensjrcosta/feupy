# Licensed under a 3-clause BSD style license - see LICENSE
"""Physical unit conversion utilities."""

import astropy.units as u

__all__ = [
    "flux_density_to_nu_fnu",
    "frequency_to_energy",
    "wavelength_to_energy",
]


def flux_density_to_nu_fnu(frequency, flux_density):
    """Convert spectral flux density to SED energy flux.

    Compute the spectral energy distribution (SED) quantity
    nu * F_nu from a flux density per unit frequency.

    The result represents the energy flux per logarithmic
    frequency interval, not the energy flux integrated over
    a finite frequency band.

    Parameters
    ----------
    frequency : `~astropy.units.Quantity`
        Observed frequency.
    flux_density : `~astropy.units.Quantity`
        Spectral flux density, e.g. Jy or mJy.

    Returns
    -------
    energy_flux : `~astropy.units.Quantity`
        SED energy flux (nu * F_nu) in erg cm-2 s-1.
    """
    frequency = frequency.to(u.Hz)
    flux_density = flux_density.to(u.Jy)

    return (frequency * flux_density).to(
        u.erg / (u.cm**2 * u.s)
    )


def frequency_to_energy(frequency):
    """Convert photon frequency to energy.

    Parameters
    ----------
    frequency : `~astropy.units.Quantity`
        Photon frequency.

    Returns
    -------
    energy : `~astropy.units.Quantity`
        Photon energy in eV.
    """
    return frequency.to(
        u.eV,
        equivalencies=u.spectral(),
    )


def wavelength_to_energy(wavelength):
    """Convert photon wavelength to energy.

    Parameters
    ----------
    wavelength : `~astropy.units.Quantity`
        Photon wavelength.

    Returns
    -------
    energy : `~astropy.units.Quantity`
        Photon energy in eV.
    """
    return wavelength.to(
        u.eV,
        equivalencies=u.spectral(),
    )