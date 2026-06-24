# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
General constants used across FeuPy.
"""

from astropy import units as u

__all__ = [
    "UNIT_DEG",
    "FRAME_ICRS",
    "FRAME_FK5",
    "CU",
]


# -----------------------------------------------------------------------------
# Coordinate constants
# -----------------------------------------------------------------------------

UNIT_DEG = "deg"
"""Default angular unit."""

FRAME_ICRS = "icrs"
"""ICRS coordinate frame."""

FRAME_FK5 = "fk5"
"""FK5 coordinate frame."""


# -----------------------------------------------------------------------------
# Physical constants
# -----------------------------------------------------------------------------

CU = 6.1e-17 * u.Unit("TeV-1 cm-2 s-1")
"""
Reference flux normalization constant.

Unit: TeV⁻¹ cm⁻² s⁻¹
"""