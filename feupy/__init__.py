# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
FEUPY: A Python package for TeV astronomy built on Gammapy.

FEUPY provides tools for:

- VHE gamma-ray data analysis
- Counterpart searches
- Multiwavelength spectral modeling
- CTAO observation simulations and sensitivity studies
- Non-thermal radiation modeling using Naima

Repository
----------
https://github.com/rubensjrcosta/feupy

Package structure (inside ``feupy/``)
-------------------------------------

    feupy/
        catalog/        Source catalog tools
        visualization/  Plotting styles and helper functions
        analysis/       High-level analysis interface

Notes
-----

- The directory ``data/`` exists in the repository but is **not**
  included in the installed Python package.
- Runtime data required by FEUPY (e.g., CTAO IRFs) must be located
  via environment variables or user-specified paths.
- Extended datasets (HAWC, LHAASO, etc.) are distributed in
  separate optional repositories.

Example
-------

>>> from feupy.catalogs import load_catalog
>>> from feupy.visualization.sed import SEDPlotter
>>> from feupy.analysis.config import CTAOAnalysisConfig
>>> from feupy.analysis.core import CTAOAnalysis
"""

import importlib
from importlib.metadata import PackageNotFoundError, version

# Package version
try:
    __version__ = version("feupy")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

# Public API
__all__ = [
    "__version__",
    "catalog",
    "visualization",
    "analysis",
]


def __getattr__(name):
    if name in ["catalog", "visualization", "analysis"]:
        return importlib.import_module(f"feupy.{name}")
    raise AttributeError(f"module 'feupy' has no attribute '{name}'")
