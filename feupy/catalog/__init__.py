# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Source catalogs."""

from .hawc import SourceCatalogObjectExtraHAWC, SourceCatalogExtraHAWC
from .lhaaso import SourceCatalogObjectPublishNatureLHAASO, SourceCatalogPublishNatureLHAASO

# from .length import get_length

# from ..scripts.example1 import yolo

# from ..scripts.make_1LHAASO import get_table

__all__ = [
    "SourceCatalogExtraHAWC",
    "SourceCatalogPublishNatureLHAASO",
    "SourceCatalogObjectExtraHAWC",
    "SourceCatalogObjectPublishNatureLHAASO",
]