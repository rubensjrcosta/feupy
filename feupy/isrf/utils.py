# Licensed under a 3-clause BSD style license - see LICENSE
"""Spectral utilities for interstellar radiation fields.

The spectral energy density convention is S(epsilon) = epsilon**2 dn/depsilon,
with units of energy per volume (e.g. eV / cm**3).
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.constants import c, k_B, sigma_sb

__all__ = [
    "integrate_isrf_sed",
    "sed_to_photon_density",
    "blackbody_sed",
    "cmb_energy_density",
    "add_cmb_to_isrf",
]

_SED_UNIT = u.eV / u.cm**3
_NUMBER_UNIT = 1 / (u.eV * u.cm**3)


def _validate_spectrum(energy: u.Quantity, sed: u.Quantity):
    """Return validated energy and SED quantities with canonical units."""
    energy = u.Quantity(energy, copy=False).to(u.eV)
    sed = u.Quantity(sed, copy=False).to(_SED_UNIT)
    if energy.ndim != 1 or energy.size < 2:
        raise ValueError("energy must be a one-dimensional array with >= 2 points")
    if sed.shape[-1] != energy.size:
        raise ValueError("last axis of sed must match energy length")
    if not np.all(np.isfinite(energy.value)) or np.any(energy.value <= 0):
        raise ValueError("energy must contain finite, positive values")
    if np.any(np.diff(energy.value) <= 0):
        raise ValueError("energy must be strictly increasing")
    if not np.all(np.isfinite(sed.value)) or np.any(sed.value < 0):
        raise ValueError("sed must contain finite, nonnegative values")
    return energy, sed


def integrate_isrf_sed(energy: u.Quantity, sed: u.Quantity) -> u.Quantity:
    """Integrate an ISRF spectral energy density over log photon energy.

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Strictly increasing photon energies.
    sed : `~astropy.units.Quantity`
        S(epsilon) = epsilon**2 dn/depsilon in energy per volume. The last
        axis must match ``energy``; preceding dimensions are preserved.

    Returns
    -------
    energy_density : `~astropy.units.Quantity`
        Integrated energy density in eV / cm**3.
    """
    energy, sed = _validate_spectrum(energy, sed)
    return np.trapezoid(sed.value, x=np.log(energy.value), axis=-1) * _SED_UNIT


def sed_to_photon_density(energy: u.Quantity, sed: u.Quantity) -> u.Quantity:
    """Convert S(epsilon) to differential photon density dn/depsilon.

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Strictly increasing photon energies.
    sed : `~astropy.units.Quantity`
        Spectral energy density in energy per volume.

    Returns
    -------
    photon_density : `~astropy.units.Quantity`
        Differential photon density in 1 / (eV cm**3).
    """
    energy, sed = _validate_spectrum(energy, sed)
    return (sed / energy**2).to(_NUMBER_UNIT)


def blackbody_sed(
    energy: u.Quantity,
    temperature: u.Quantity,
    energy_density: u.Quantity,
) -> u.Quantity:
    """Evaluate a normalized Planck spectral energy density.

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Photon energies, strictly positive.
    temperature : `~astropy.units.Quantity`
        Positive blackbody temperature.
    energy_density : `~astropy.units.Quantity`
        Bolometric energy density.

    Returns
    -------
    sed : `~astropy.units.Quantity`
        S(epsilon) in eV / cm**3.

    Notes
    -----
    The spectral shape is (15/pi**4) x**4 / (exp(x) - 1), x=epsilon/kT.
    Its integral over d ln(epsilon) is unity on (0, infinity).
    """
    energy = u.Quantity(energy, copy=False).to(u.eV)
    temperature = u.Quantity(temperature, copy=False).to(u.K)
    energy_density = u.Quantity(energy_density, copy=False).to(_SED_UNIT)
    if np.any(~np.isfinite(energy.value)) or np.any(energy.value <= 0):
        raise ValueError("energy must be finite and positive")
    if np.any(~np.isfinite(temperature.value)) or np.any(temperature.value <= 0):
        raise ValueError("temperature must be finite and positive")
    if np.any(~np.isfinite(energy_density.value)) or np.any(energy_density.value < 0):
        raise ValueError("energy_density must be finite and nonnegative")
    x = (energy / (k_B * temperature).to(u.eV)).to_value(u.one)
    shape = np.zeros_like(x, dtype=float)
    mask = x < 100.0
    shape[mask] = (15.0 / np.pi**4) * x[mask] ** 4 / np.expm1(x[mask])
    return energy_density * shape


def cmb_energy_density(temperature: u.Quantity = 2.7255 * u.K) -> u.Quantity:
    """Return the bolometric energy density of a blackbody CMB."""
    temperature = u.Quantity(temperature, copy=False).to(u.K)
    if np.any(~np.isfinite(temperature.value)) or np.any(temperature.value <= 0):
        raise ValueError("temperature must be finite and positive")
    return (4 * sigma_sb / c * temperature**4).to(_SED_UNIT)


def add_cmb_to_isrf(
    energy: u.Quantity,
    sed: u.Quantity,
    temperature: u.Quantity = 2.7255 * u.K,
) -> u.Quantity:
    """Add a CMB blackbody sampled on an existing photon-energy grid.

    Notes
    -----
    The input ISRF must exclude the CMB. The returned spectrum is truncated
    to the supplied energy grid, so integrating it may miss CMB energy
    outside that grid. Use ``cmb_energy_density`` for its bolometric value.
    """
    energy, sed = _validate_spectrum(energy, sed)
    return sed + blackbody_sed(energy, temperature, cmb_energy_density(temperature))
