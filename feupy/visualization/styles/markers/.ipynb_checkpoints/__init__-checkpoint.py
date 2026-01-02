"""Plotting styles for feupy."""

from .palettes import (
    PALETTE_DEFAULT,
    PALETTE_TABLEAU,
    PALETTE_IBM,
    PALETTE_WONG,
)

from .linestyles import LINESTYLES_DEFAULT

from .markers import (
    generate_catalog_markers,
    generate_specified_marker_set,
    get_kwargs_fit,
    write_ref_markers,
    read_ref_markers,
)

__all__ = [
    "PALETTE_DEFAULT",
    "PALETTE_TABLEAU",
    "PALETTE_IBM",
    "PALETTE_WONG",
    "LINESTYLES_DEFAULT",
    "generate_catalog_markers",
    "generate_specified_marker_set",
    "get_kwargs_fit",
    "write_ref_markers",
    "read_ref_markers",
]
