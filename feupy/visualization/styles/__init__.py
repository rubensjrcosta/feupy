# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Styles tools."""

from importlib.resources import files

# Palettes
from .palettes import (
    PALETTE_DEFAULT,
    PALETTE_TABLEAU,
    PALETTE_IBM,
    PALETTE_WONG,
)

# Line styles
from .linestyles import (
    LINESTYLES_DEFAULT,
)

# Marker styles
from .markers import (
    CatalogStyle,
    CatalogStyleRegistry,
    CATALOG_STYLE_REGISTRY,
    resolve_marker,
    resolve_marker_size,
    build_point_kwargs,
    build_fp_kwargs,
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