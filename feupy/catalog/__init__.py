# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Source catalogs."""

from .registry import FEUPY_CATALOG_REGISTRY

# Re-export catalog classes so the API stays the same
from .hawc import (
    SourceCatalogObjectEHWC, SourceCatalogEHWC,
    SourceCatalogObjectExtraHAWC, SourceCatalogExtraHAWC,
)
from .hess import (
    SourceCatalogObjectExtraHESS, SourceCatalogExtraHESS,
)
from .veritas import (
    SourceCatalogVTSCat, SourceCatalogObjectVTSCat,
    SourceCatalogVERITAS, SourceCatalogObjectVERITAS,
)
from .psrcat import (
    SourceCatalogPSRCAT, SourceCatalogObjectPSRCAT,
)
from .lhaaso import (
    SourceCatalogObjectLHAASO, SourceCatalogLHAASO,
    SourceCatalogObjectExtraLHAASO, SourceCatalogExtraLHAASO,
)

__all__ = [
    "FEUPY_CATALOG_REGISTRY",
    "SourceCatalogVTSCat",
    "SourceCatalogObjectVTSCat",
    "SourceCatalogVERITAS",
    "SourceCatalogObjectVERITAS",
    "SourceCatalogPSRCAT",
    "SourceCatalogObjectPSRCAT",
    "SourceCatalogObjectLHAASO",
    "SourceCatalogLHAASO",
    "SourceCatalogExtraLHAASO",
    "SourceCatalogObjectExtraLHAASO",
    "SourceCatalogObjectEHWC",
    "SourceCatalogObjectExtraHAWC",
    "SourceCatalogExtraHAWC",
    "SourceCatalogEHWC",
    "SourceCatalogObjectExtraHESS",
    "SourceCatalogExtraHESS",
]