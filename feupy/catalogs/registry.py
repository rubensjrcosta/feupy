# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Catalog registry for FeuPy."""

import os
from pathlib import Path

from gammapy.catalog import CATALOG_REGISTRY
from gammapy.utils.registry import Registry

FEUPY_DATA = Path(os.environ.get("FEUPY_DATA", ""))
HAS_FEUPY_DATASETS = bool(os.environ.get("FEUPY_DATA")) and FEUPY_DATA.exists()

FEUPY_CATALOGS = list(CATALOG_REGISTRY)

if HAS_FEUPY_DATASETS:
    from .hawc import SourceCatalogEHWC, SourceCatalogExtraHAWC
    from .hess import SourceCatalogExtraHESS
    from .lhaaso import SourceCatalogExtraLHAASO, SourceCatalogLHAASO
    from .psrcat import SourceCatalogPSRCAT
    from .veritas import SourceCatalogVERITASCygnus, SourceCatalogVTSCat

    FEUPY_CATALOGS.extend(
        [
            SourceCatalogEHWC,
            SourceCatalogExtraHAWC,
            SourceCatalogExtraHESS,
            SourceCatalogVTSCat,
            SourceCatalogVERITASCygnus,
            SourceCatalogPSRCAT,
            SourceCatalogLHAASO,
            SourceCatalogExtraLHAASO,
        ]
    )

FEUPY_CATALOG_REGISTRY = Registry(FEUPY_CATALOGS)

__all__ = [
    "FEUPY_CATALOG_REGISTRY",
    "HAS_FEUPY_DATASETS",
]
