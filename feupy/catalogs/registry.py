# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""catalog registry."""
import os
from pathlib import Path
from gammapy.catalog import CATALOG_REGISTRY
from gammapy.utils.registry import Registry


# ---------------------------------------------------------
# Detect optional dataset package
# ---------------------------------------------------------

FEUPY_DATA = Path(os.environ.get("FEUPY_DATA", ""))
HAS_FEUPY_DATASETS = FEUPY_DATA.exists()


# ---------------------------------------------------------
# Base registry (Gammapy catalogs always available)
# ---------------------------------------------------------

FEUPY_CATALOGS = list(CATALOG_REGISTRY)

# ---------------------------------------------------------
# Optional FEUPY catalogs (only if datasets exist)
# ---------------------------------------------------------

if HAS_FEUPY_DATASETS:

    from .hawc import (
        SourceCatalogEHWC,
        SourceCatalogExtraHAWC,
    )

    from .hess import SourceCatalogExtraHESS

    from .veritas import (
        SourceCatalogVTSCat,
        SourceCatalogVERITASCygnus,
    )

    from .psrcat import SourceCatalogPSRCAT

    from .lhaaso import (
        SourceCatalogLHAASO,
        SourceCatalogExtraLHAASO,
    )

    FEUPY_CATALOGS.extend([
        SourceCatalogEHWC,
        SourceCatalogExtraHAWC,
        SourceCatalogExtraHESS,
        SourceCatalogVTSCat,
        SourceCatalogVERITASCygnus,
        SourceCatalogPSRCAT,
        SourceCatalogLHAASO,
        SourceCatalogExtraLHAASO,
    ])


# ---------------------------------------------------------
# Final registry object
# ---------------------------------------------------------

FEUPY_CATALOG_REGISTRY = Registry(FEUPY_CATALOGS)


__all__ = [
    "FEUPY_CATALOG_REGISTRY",
    "HAS_FEUPY_DATASETS",
]