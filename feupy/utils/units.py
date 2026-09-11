# Licensed under a 3-clause BSD style license - see LICENSE
"""Unit helper utilities."""

from astropy import units as u

__all__ = ["is_energy_unit"]


def is_energy_unit(unit):
    """Check whether a unit is equivalent to an energy unit.

    Parameters
    ----------
    unit : `~astropy.units.Unit` or str
        Unit to check.

    Returns
    -------
    bool
        True if the unit is equivalent to energy, otherwise False.
    """
    try:
        u.Unit(unit).to(u.eV)
    except (TypeError, ValueError, u.UnitConversionError):
        return False

    return True
