# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Units helper utilities.
"""

from astropy import units as u

__all__ = [
    "is_energy_unit",
]


def is_energy_unit(unit):
    """
    Check if unit is an energy unit.

    Parameters
    ----------
    unit : `~astropy.units.Unit`

    Returns
    -------
    bool
    """

    try:
        (1 * unit).to(u.eV)
        return True
    except Exception:
        return False