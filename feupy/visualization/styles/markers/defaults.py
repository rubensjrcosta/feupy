# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Default marker styles for catalogs."""

from .registry import CatalogStyle, CATALOG_STYLE_REGISTRY

__all__ = ["DEFAULT_MARKERS"]


DEFAULT_MARKERS = {
    "psrcat": "*",
    "2pc": "*",
    "3pc": "*",
    "gamma-cat": "h",
    "hgps": "p",
    "hess-2019a&a": "p",
    "3fgl": "v",
    "4fgl": "v",
    "2fhl": "v",
    "3fhl": "v",
    "2hwc": "8",
    "3hwc": "8",
    "ehwc": "8",
    "hwc-2021apj": "8",
    "veritas-2018apj": ">",
    "vtscat": ">",
    "1lhaaso": "s",
    "lhaaso": "s",
}


# Register defaults in the global registry
for tag, marker in DEFAULT_MARKERS.items():
    CATALOG_STYLE_REGISTRY.register(
        CatalogStyle(tag=tag, marker=marker)
    )