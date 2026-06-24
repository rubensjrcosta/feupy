# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Marker styles."""

from .registry import CatalogStyle, CatalogStyleRegistry, CATALOG_STYLE_REGISTRY
from .defaults import DEFAULT_MARKERS
from .resolver import extract_catalog_tag, resolve_marker
from .marker_size import MARKERS_DEFAULT_DICT, resolve_marker_size
from .plotting import build_point_kwargs, build_fp_kwargs


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