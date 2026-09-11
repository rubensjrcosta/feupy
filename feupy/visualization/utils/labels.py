# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Visualization labels for FeuPy."""

from astropy import units as u
from gammapy.estimators.map.core import DEFAULT_UNIT
from gammapy.maps.axes import UNIT_STRING_FORMAT

__all__ = [
    "DEFAULT_YAXIS_LABEL",
    "DEFAULT_XAXIS_LABEL",
]

# ---------------------------------------------------------------------
# Default Y-axis labels for SED plots
# ---------------------------------------------------------------------
DEFAULT_YAXIS_LABEL = {
    "e2dnde": (
        f"[{DEFAULT_UNIT['e2dnde'].to_string(UNIT_STRING_FORMAT)}]".replace(
            "[$\\mathrm{", "$\\rm {E^{2}\\,\\Phi(E)\\, ["
        )
    ),
    "dnde": (
        f"[{DEFAULT_UNIT['dnde'].to_string(UNIT_STRING_FORMAT)}]".replace(
            "[$\\mathrm{", "$\\rm {\\Phi(E)\\, ["
        )
    ),
}

# ---------------------------------------------------------------------
# Default X-axis labels for energy axes
# ---------------------------------------------------------------------
DEFAULT_XAXIS_LABEL = {
    "erg": f"Energy [{u.Unit('erg').to_string(UNIT_STRING_FORMAT)}]",
    "TeV": f"Energy [{u.Unit('TeV').to_string(UNIT_STRING_FORMAT)}]",
}
