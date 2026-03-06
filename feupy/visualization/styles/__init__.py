# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Plotting styles for feupy.

This subpackage provides:
- Matplotlib style sheets (.mplstyle)
- Color palettes
- Line styles
- Marker utilities

The public API is intentionally small and stable, following the
Gammapy and Astropy visualization design.
"""

from importlib.resources import files

# ============================================================================
# Color palettes
# ============================================================================

from .palettes import (  # noqa: F401
    PALETTE_DEFAULT,
    PALETTE_TABLEAU,
    PALETTE_IBM,
    PALETTE_WONG,
)

# ============================================================================
# Line styles
# ============================================================================

from .linestyles import LINESTYLES_DEFAULT  # noqa: F401

# ============================================================================
# Marker utilities (public API only)
# ============================================================================

from .markers import (  # noqa: F401
    map_catalog_to_marker,
    make_marker_dict,
    get_fit_plot_kwargs,
)

# ============================================================================
# Matplotlib style
# ============================================================================

FEUPY_MPL_STYLE = files(__name__) / "mystyle.mplstyle"

# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Matplotlib style
    "FEUPY_MPL_STYLE",
    # Palettes
    "PALETTE_DEFAULT",
    "PALETTE_TABLEAU",
    "PALETTE_IBM",
    "PALETTE_WONG",
    # Line styles
    "LINESTYLES_DEFAULT",
    # Marker helpers
    "map_catalog_to_marker",
    "make_marker_dict",
    "get_fit_plot_kwargs",
]
