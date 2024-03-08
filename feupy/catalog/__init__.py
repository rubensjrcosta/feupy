# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Source catalogs."""

from gammapy.utils.registry import Registry
from gammapy.catalog import CATALOG_REGISTRY as CATALOG_REGISTRY_GAMMAPY

from .gamma.hawc import SourceCatalogObjectExtraHAWC, SourceCatalogExtraHAWC
from .gamma.lhaaso import SourceCatalogObjectPublishNatureLHAASO, SourceCatalogPublishNatureLHAASO
from .pulsar.atnf import SourceCatalogObjectATNF, SourceCatalogATNF

__all__ = [
    "SourceCatalogExtraHAWC",
    "SourceCatalogPublishNatureLHAASO",
    "SourceCatalogATNF",
    "SourceCatalogObjectExtraHAWC",
    "SourceCatalogObjectPublishNatureLHAASO",
    "SourceCatalogObjectATNF",
]

_cats = list(CATALOG_REGISTRY_GAMMAPY)

_feupy_cats = [
        SourceCatalogExtraHAWC,
        SourceCatalogPublishNatureLHAASO,
        SourceCatalogATNF,
    ]

CATALOG_REGISTRY_FEUPY = Registry(_feupy_cats)

_cats.extend(CATALOG_REGISTRY_FEUPY)

CATALOG_REGISTRY = Registry(_cats)

"""Registry of source catalogs in FeuPy."""
