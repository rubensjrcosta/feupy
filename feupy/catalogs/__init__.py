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
    from .lhaaso import (
        SourceCatalogExtraLHAASO,
        SourceCatalogLHAASO,
        SourceCatalogObjectExtraLHAASO,
        SourceCatalogObjectLHAASO,
    )
    from .psrcat import (
        SourceCatalogObjectPSRCAT,
        SourceCatalogPSRCAT,
    )
    from .veritas import (
        SourceCatalogObjectVERITASCygnus,
        SourceCatalogObjectVTSCat,
        SourceCatalogVERITASCygnus,
        SourceCatalogVTSCat,
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
