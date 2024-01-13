# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Source catalogs."""

from .hawc import SourceCatalogObjectExtraHAWC, SourceCatalogExtraHAWC
from .lhaaso import SourceCatalogObjectPublishNatureLHAASO, SourceCatalogPublishNatureLHAASO
from .pulsar.atnf import SourceCatalogObjectATNF, SourceCatalogATNF

__all__ = [
    "SourceCatalogExtraHAWC",
    "SourceCatalogPublishNatureLHAASO",
    "SourceCatalogATNF",
    "SourceCatalogObjectExtraHAWC",
    "SourceCatalogObjectPublishNatureLHAASO",
    "SourceCatalogObjectATNF",
]