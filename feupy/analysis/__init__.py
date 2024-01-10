# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Analysis."""
# from .maker import get_catalogs, gammapy_catalog, atnf_catalog
from .counterparts import AnalysisConfig, Analysis

__all__ = [
    "Analysis",
    "AnalysisConfig",
#     "get_catalogs",
#     "gammapy_catalog",
#     "atnf_catalog",
]