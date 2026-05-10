# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
SED table column definitions and default units.
"""

from astropy import units as u

__all__ = [
    "ENERGY_COLUMNS",
    "SED_COLUMNS",
    "DEFAULT_ENERGY_UNIT",
    "DEFAULT_SED_UNIT",
]


# Required energy columns for different SED types
ENERGY_COLUMNS = {
    "dnde": ["e_ref"],
    "e2dnde": ["e_ref"],
    "flux": ["e_min", "e_max", "flux_err"],
    "eflux": ["e_min", "e_max", "eflux_err"],
}


# Required SED value columns
SED_COLUMNS = {
    "dnde": ["dnde", "dnde_err", "dnde_ul"],
    "e2dnde": ["e2dnde", "e2dnde_err", "e2dnde_ul"],
}


# Default units for energy axes
DEFAULT_ENERGY_UNIT = {
    "dnde": u.TeV,
    "e2dnde": u.TeV,
}


# Default units for SED values
DEFAULT_SED_UNIT = {
    "dnde": u.Unit("cm-2 s-1 TeV-1"),
    "e2dnde": u.Unit("erg cm-2 s-1"),
}