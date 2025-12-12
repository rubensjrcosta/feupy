# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
FeuPy: A Python package for TeV Astronomy built on Gammapy.

FeuPy provides tools for:
- VHE gamma-ray data analysis
- Counterpart searches
- Multiwavelength spectral modeling
- CTAO observation simulations and sensitivity studies
- Non-thermal radiation modeling using Naima

Repository:
    https://github.com/rubensjrcosta/feupy

Package structure (inside ``feupy/``):

    feupy/
        catalog/        --- Source catalog tools
        visualization/  --- Plotting styles and helper functions
        analysis/         --- Feupy high level interface (analysis)

Notes:
    - The directory ``data/`` exists in the repository,
      but is **not** part of the installed Python package.
    - Runtime data required by FeuPy (e.g., CTAO IRFs) must be located
      either via environment variables or user-specified paths.
    - Additional extended datasets (HAWC, LHAASO etc.) are provided
      in separate external repositories and are optional.

Example usage:

    >>> from feupy.catalog import load_catalog
    >>> from feupy.visualization.sed import SEDPlotter
    >>> from feupy.analysis.config import CTAOAnalysisConfig
    >>> from feupy.analysis.core import CTAOAnalysis
"""

from . import catalog
from . import visualization
from . import analysis

__all__ = ["catalog", "visualization", "analysis"]
