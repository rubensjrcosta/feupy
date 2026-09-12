# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FeuPy: Tools for very-high-energy gamma-ray astronomy.

FeuPy provides utilities for gamma-ray data analysis, source catalog
handling, multiwavelength spectral modeling, CTAO simulations,
sensitivity studies, and non-thermal radiation modeling.

Main subpackages
----------------
analysis
    High-level analysis and simulation tools.
catalogs
    Source catalog and counterpart utilities.
core
    Core source and data structures.
irf
    Instrument response function utilities.
naima
    Non-thermal particle and radiation modeling tools.
visualization
    Plotting functions and visualization styles.

Notes
-----
Runtime data such as CTAO instrument response functions are not distributed
with the Python package and must be made available through the appropriate
data paths or environment variables.

Examples
--------
>>> from feupy.analysis.config import CTAOAnalysisConfig
>>> from feupy.analysis.core import CTAOAnalysis
>>> from feupy.catalogs import load_catalog
"""

import importlib
from importlib.metadata import PackageNotFoundError, version

__all__ = [
    "__version__",
    "analysis",
    "catalogs",
    "core",
    "irf",
    "naima",
    "visualization",
]


try:
    __version__ = version("feupy")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


def __getattr__(name):
    """Lazily import public FeuPy subpackages."""
    if name in __all__[1:]:
        return importlib.import_module(f"feupy.{name}")

    raise AttributeError(f"module 'feupy' has no attribute {name!r}")
