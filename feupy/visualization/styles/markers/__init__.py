# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Marker styles."""

from .defaults import DEFAULT_MARKERS
from .marker_size import MARKERS_DEFAULT_DICT, resolve_marker_size
from .plotting import build_fp_kwargs, build_point_kwargs
from .registry import CATALOG_STYLE_REGISTRY, CatalogStyle, CatalogStyleRegistry
from .resolver import extract_catalog_tag, resolve_marker

__all__ = [
    "CatalogStyle",
    "CatalogStyleRegistry",
    "CATALOG_STYLE_REGISTRY",
    "DEFAULT_MARKERS",
    "extract_catalog_tag",
    "resolve_marker",
    "MARKERS_DEFAULT_DICT",
    "resolve_marker_size",
    "build_point_kwargs",
    "build_fp_kwargs",
]
