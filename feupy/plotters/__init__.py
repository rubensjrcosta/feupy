# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Plotters."""
from .seting import set_leg_style, set_leg_style_models, set_leg_style_datasets
from .spectral_energy_distribution import show_SED
from .sky_map import show_sky_map

__all__ = [
    "set_leg_style",
    "set_leg_style_models",
    "set_leg_style_datasets",
    "show_SED",
    "show_sky_map",
]