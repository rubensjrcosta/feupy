# Licensed under a 3-clause BSD style license - see LICENSE
"""SED table column definitions and default units."""

from astropy import units as u

__all__ = [
    "ENERGY_COLUMNS",
    "SED_COLUMNS",
    "DEFAULT_ENERGY_UNIT",
    "DEFAULT_SED_UNIT",
]

ENERGY_COLUMNS = {
    "dnde": ["e_ref"],
    "e2dnde": ["e_ref"],
    "flux": ["e_min", "e_max", "flux_err"],
    "eflux": ["e_min", "e_max", "eflux_err"],
}
"""Required energy-related columns for each supported SED type."""

SED_COLUMNS = {
    "dnde": ["dnde", "dnde_err", "dnde_ul"],
    "e2dnde": ["e2dnde", "e2dnde_err", "e2dnde_ul"],
}
"""Required SED value columns for each supported differential SED type."""

DEFAULT_ENERGY_UNIT = {
    "dnde": u.TeV,
    "e2dnde": u.TeV,
}
"""Default energy units for differential SED types."""

DEFAULT_SED_UNIT = {
    "dnde": u.Unit("cm-2 s-1 TeV-1"),
    "e2dnde": u.Unit("erg cm-2 s-1"),
}
"""Default physical units for differential SED values."""
