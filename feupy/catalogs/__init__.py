# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Source catalogs API."""

from .registry import FEUPY_CATALOG_REGISTRY, HAS_FEUPY_DATASETS

# ---------------------------------------------------------
# Base API (always available)
# ---------------------------------------------------------

__all__ = [
    "FEUPY_CATALOG_REGISTRY",
    "HAS_FEUPY_DATASETS",
]

# ---------------------------------------------------------
# Optional dataset catalogs
# ---------------------------------------------------------

if HAS_FEUPY_DATASETS:
    from .hawc import (
        SourceCatalogEHWC,
        SourceCatalogExtraHAWC,
        SourceCatalogObjectEHWC,
        SourceCatalogObjectExtraHAWC,
    )
    from .hess import (
        SourceCatalogExtraHESS,
        SourceCatalogObjectExtraHESS,
    )
    from .veritas import (
        SourceCatalogVTSCat,
        SourceCatalogVERITASCygnus,
        SourceCatalogObjectVTSCat,
        SourceCatalogObjectVERITASCygnus,
    )
    from .psrcat import (
        SourceCatalogPSRCAT,
        SourceCatalogObjectPSRCAT,
    )
    from .lhaaso import (
        SourceCatalogLHAASO,
        SourceCatalogExtraLHAASO,
        SourceCatalogObjectLHAASO,
        SourceCatalogObjectExtraLHAASO,
    )

    __all__ += [
        "SourceCatalogEHWC",
        "SourceCatalogExtraHAWC",
        "SourceCatalogObjectEHWC",
        "SourceCatalogObjectExtraHAWC",
        "SourceCatalogExtraHESS",
        "SourceCatalogObjectExtraHESS",
        "SourceCatalogVTSCat",
        "SourceCatalogVERITASCygnus",
        "SourceCatalogObjectVTSCat",
        "SourceCatalogObjectVERITASCygnus",
        "SourceCatalogPSRCAT",
        "SourceCatalogObjectPSRCAT",
        "SourceCatalogLHAASO",
        "SourceCatalogExtraLHAASO",
        "SourceCatalogObjectLHAASO",
        "SourceCatalogObjectExtraLHAASO",
    ]