# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Default axis labels used by FeuPy visualization utilities."""

from astropy import units as u
from gammapy.estimators.map.core import DEFAULT_UNIT
from gammapy.maps.axes import UNIT_STRING_FORMAT

__all__ = [
    "DEFAULT_XAXIS_LABEL",
    "DEFAULT_YAXIS_LABEL",
]


DEFAULT_YAXIS_LABEL = {
    "e2dnde": (
        f"[{DEFAULT_UNIT['e2dnde'].to_string(UNIT_STRING_FORMAT)}]".replace(
            "[$\\mathrm{",
            "$\\rm {E^{2}\\,\\Phi(E)\\, [",
        )
    ),
    "dnde": (
        f"[{DEFAULT_UNIT['dnde'].to_string(UNIT_STRING_FORMAT)}]".replace(
            "[$\\mathrm{",
            "$\\rm {\\Phi(E)\\, [",
        )
    ),
}

DEFAULT_XAXIS_LABEL = {
    unit: f"Energy [{u.Unit(unit).to_string(UNIT_STRING_FORMAT)}]"
    for unit in ("erg", "TeV")
}
