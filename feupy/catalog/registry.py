# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""catalog registry."""
from gammapy.catalog import CATALOG_REGISTRY
from gammapy.utils.registry import Registry

from .hawc import (
    SourceCatalogObjectEHWC, SourceCatalogEHWC,
    SourceCatalogObjectExtraHAWC, SourceCatalogExtraHAWC
)
from .hess import (
    SourceCatalogObjectExtraHESS, SourceCatalogExtraHESS
)
from .veritas import (
    SourceCatalogVTSCat, SourceCatalogObjectVTSCat,
    SourceCatalogVERITAS, SourceCatalogObjectVERITAS
)
from .psrcat import (
    SourceCatalogPSRCAT, SourceCatalogObjectPSRCAT
)
from .lhaaso import (
    SourceCatalogObjectLHAASO, SourceCatalogLHAASO,
    SourceCatalogObjectExtraLHAASO, SourceCatalogExtraLHAASO
)

# Combine registries
GAMMAPY_CATALOGS = CATALOG_REGISTRY.copy()

FEUPY_CATALOGS = GAMMAPY_CATALOGS + [
    SourceCatalogEHWC,
    SourceCatalogExtraHAWC,
    SourceCatalogExtraHESS,
    SourceCatalogVTSCat,
    SourceCatalogVERITAS,
    SourceCatalogLHAASO,
    SourceCatalogExtraLHAASO,
    SourceCatalogPSRCAT,
]

FEUPY_CATALOG_REGISTRY = Registry(FEUPY_CATALOGS)
