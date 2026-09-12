# Licensed under a 3-clause BSD style license - see LICENSE
"""Formatting utilities for filenames and scientific tags."""

import re

from astropy import units as u

__all__ = [
    "string_to_filename",
    "energy_to_string",
    "energy_range_to_string",
]


def string_to_filename(name, strict=False):
    """Convert a string to a filename-safe format.

    Parameters
    ----------
    name : str
        Input string.
    strict : bool, optional
        Whether to remove characters outside the allowed set.
        Default is False.

    Returns
    -------
    filename : str
        Filename-safe string.

    Notes
    -----
    When ``strict=True``, only alphanumeric characters and ``_``, ``-``,
    ``+``, ``*``, and ``.`` are preserved.
    """
    name = re.sub(r"\s+", "_", name)

    if strict:
        name = re.sub(r"[^\w+*.-]", "", name)

    return name


def energy_to_string(energy, unit=None):
    """Convert an energy quantity to a compact string.

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Energy value.
    unit : `~astropy.units.Unit`, optional
        Target unit. If not given, an appropriate unit is selected
        automatically.

    Returns
    -------
    energy_string : str
        Compact energy string.

    Examples
    --------
    >>> energy_to_string(2 * u.TeV)
    '2TeV'
    """
    if unit is not None:
        energy = energy.to(unit)
    elif energy >= 1 * u.PeV:
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


def energy_range_to_string(energy_min, energy_max):
    """Convert an energy range to a compact string.

    Parameters
    ----------
    energy_min : `~astropy.units.Quantity`
        Minimum energy.
    energy_max : `~astropy.units.Quantity`
        Maximum energy.

    Returns
    -------
    energy_range : str
        Compact energy-range string.

    Examples
    --------
    >>> energy_range_to_string(100 * u.GeV, 10 * u.TeV)
    '100GeV_10TeV'
    """
    return f"{energy_to_string(energy_min)}_{energy_to_string(energy_max)}"
