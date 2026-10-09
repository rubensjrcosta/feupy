# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Shared pytest fixtures for FeuPy tests."""

import os
from pathlib import Path

import pytest


@pytest.fixture
def require_prod6_data():
    """Require the external CTAO Prod6 FITS dataset."""

    data_root = os.environ.get("FEUPY_DATA")

    if not data_root:
        pytest.skip("FEUPY_DATA is not configured.")

    prod6_path = (
        Path(data_root)
        / "irfs"
        / "ctao-prod6-zenodo-v1.0"
        / "fits"
    )

    if not prod6_path.is_dir():
        pytest.skip("CTAO Prod6 FITS data are not available.")

    if not any(prod6_path.rglob("*.fits.gz")):
        pytest.skip("CTAO Prod6 FITS files are not available.")

    return prod6_path