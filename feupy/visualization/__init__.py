# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Visualization tools."""

from .styles import (
    CatalogStyle,
    CatalogStyleRegistry,
    CATALOG_STYLE_REGISTRY,
    resolve_marker,
    resolve_marker_size,
    build_point_kwargs,
    build_fp_kwargs,
    PALETTE_DEFAULT,
    PALETTE_TABLEAU,
    PALETTE_IBM,
    PALETTE_WONG,
    LINESTYLES_DEFAULT,
    FEUPY_MPL_STYLE,
)

__all__ = [
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
]