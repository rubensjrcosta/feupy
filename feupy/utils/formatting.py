# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Formatting utilities for filenames and scientific tags."""

from __future__ import annotations

import re
from astropy import units as u

__all__ = [
    "string_to_filename",
    "energy_to_string",
    "energy_range_to_string",
]


def string_to_filename(name):
    """
    Convert a string to a filename-safe format.

    Parameters
    ----------
    name : str
        Input string.

    Returns
    -------
    str
        Filename-safe string.
    """
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w]", "", name)
    return name
    
def energy_to_string(energy, unit=None):
    """
    Convert energy to compact string.

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Energy value.
    unit : `~astropy.units.Unit`, optional
        Target unit. If None an appropriate unit is chosen automatically.

    Returns
    -------
    str
        Energy tag string.

    Examples
    --------
    >>> energy_to_tag(2 * u.TeV)
    '2TeV'
    """

    if unit is not None:
        energy = energy.to(unit)

    else:
        if energy >= 1 * u.PeV:
            energy = energy.to(u.PeV)
        elif energy >= 1 * u.TeV:
            energy = energy.to(u.TeV)
        elif energy >= 1 * u.GeV:
            energy = energy.to(u.GeV)
        elif energy >= 1 * u.MeV:
            energy = energy.to(u.MeV)

    value = energy.value

    if float(value).is_integer():
        value = int(value)

    return f"{value}{energy.unit.to_string()}"

def energy_range_to_string(Emin, Emax):
    """
    Format energy range for tags.

    Example
    -------
    100 GeV – 10 TeV -> "100GeV_10TeV"
    """
    return f"{energy_to_string(Emin)}_{energy_to_string(Emax)}"