# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Visualization tools."""

# Import core styles and plotting helpers
from .styles import (
    CATALOG_STYLE_REGISTRY,
    FEUPY_MPL_STYLE,
    LINESTYLES_DEFAULT,
    PALETTE_DEFAULT,
    PALETTE_IBM,
    PALETTE_TABLEAU,
    PALETTE_WONG,
    CatalogStyle,
    CatalogStyleRegistry,
    build_fp_kwargs,
    build_point_kwargs,
    resolve_marker,
    resolve_marker_size,
)

# Import submodules explicitly
from .utils import labels

# Expose public API
__all__ = [
    # Styles
    "CatalogStyle",
    "CatalogStyleRegistry",
    "CATALOG_STYLE_REGISTRY",
    "resolve_marker",
    "resolve_marker_size",
    "build_point_kwargs",
    "build_fp_kwargs",
    "PALETTE_DEFAULT",
    "PALETTE_TABLEAU",
    "PALETTE_IBM",
    "PALETTE_WONG",
    "LINESTYLES_DEFAULT",
    "FEUPY_MPL_STYLE",
    # Labels
    "labels",
]
