# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Visualization tools and styles for feupy."""

from importlib.resources import files

# ============================================================================
# Public visualization helpers
# ============================================================================

from .styles import (  # noqa: F401
    # Palettes
    PALETTE_DEFAULT,
    PALETTE_TABLEAU,
    PALETTE_IBM,
    PALETTE_WONG,
    # Line styles
    LINESTYLES_DEFAULT,
    # Marker helpers
    map_catalog_to_marker,
    make_marker_dict,
    get_fit_plot_kwargs,
)

# ============================================================================
# Matplotlib style
# ============================================================================

FEUPY_MPL_STYLE = files("feupy.visualization.styles") / "mystyle.mplstyle"

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
