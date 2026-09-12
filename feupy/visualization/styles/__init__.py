# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Styles tools."""

from importlib.resources import files

# Line styles
from .linestyles import (
    LINESTYLES_DEFAULT,
)

# Marker styles
from .markers import (
    CATALOG_STYLE_REGISTRY,
    CatalogStyle,
    CatalogStyleRegistry,
    build_fp_kwargs,
    build_point_kwargs,
    resolve_marker,
    resolve_marker_size,
)

# Palettes
from .palettes import (
    PALETTE_DEFAULT,
    PALETTE_IBM,
    PALETTE_TABLEAU,
    PALETTE_WONG,
)

# Matplotlib style
FEUPY_MPL_STYLE = str(files(__name__) / "mystyle.mplstyle")

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
