# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Feupy integration and system tests.

This package can be used for tests that involved several
Feupy sub-packages, or that don't fit anywhere else.
"""
from .tests import (
    test_target,
    test_roi,
test_cta_obs_parm)

__all__ = [
    "test_target",
    "test_roi",
    "test_cta_obs_parm"
]