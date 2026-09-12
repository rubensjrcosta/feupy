# Licensed under a 3-clause BSD style license - see LICENSE.rst

import astropy.units as u
import pytest
from astropy.table import Table

from feupy.catalogs.psrcat import (
    SourceCatalogObjectPSRCAT,
    SourceCatalogPSRCAT,
)


@pytest.fixture
def psrcat_file(tmp_path):
    table = Table()

    table["NAME"] = ["J0000+0000"]
    table["RAJ2000"] = [10.0]
    table["RAJ2000_ERR"] = [0.1]
    table["DEJ2000"] = [-20.0]
    table["DEJ2000_ERR"] = [0.2]
    table["P0"] = [0.5] * u.s
    table["P0_ERR"] = [0.01]
    table["DIST"] = [1.5]
    table["DIST_DM"] = [1.7]
    table["ASSOC"] = ["TEST"]
    table["TYPE"] = ["PSR"]
    table["AGE"] = [1.0e5]
    table["BSURF"] = [2.0e12]
    table["EDOT"] = [3.0e35]

    for name in table.colnames:
        table[name].description = f"Description for {name}"

    filename = tmp_path / "psrcat.fits"
    table.write(filename, format="fits")

    return filename


def test_catalog_initialization(psrcat_file):
    catalog = SourceCatalogPSRCAT(filename=psrcat_file)

    assert catalog.tag == "psrcat"
    assert len(catalog.table) == 1
    assert catalog.table["NAME"][0] == "J0000+0000"
    assert catalog.source_object_class is SourceCatalogObjectPSRCAT


def test_psr_params(psrcat_file):
    catalog = SourceCatalogPSRCAT(filename=psrcat_file)

    assert catalog.PSR_PARAMS == catalog.table.colnames
    assert "NAME" in catalog.PSR_PARAMS
    assert "P0" in catalog.PSR_PARAMS


def test_psr_params_description(psrcat_file):
    catalog = SourceCatalogPSRCAT(filename=psrcat_file)

    description = catalog.PSR_PARAMS_DESCRIPTION

    assert "*** The Pulsar Parameters ***" in description
    assert "NAME:" in description
    assert "P0:" in description


def test_source_info_all(psrcat_file):
    catalog = SourceCatalogPSRCAT(filename=psrcat_file)
    source = catalog["J0000+0000"]

    info = source.info()

    assert "*** Basic info ***" in info
    assert "*** Position info ***" in info
    assert "*** Timing and profile info ***" in info
    assert "*** Distance info ***" in info
    assert "*** Associations and survey info ***" in info
    assert "*** Derived parameters info ***" in info


def test_source_info_selected_sections(psrcat_file):
    catalog = SourceCatalogPSRCAT(filename=psrcat_file)
    source = catalog["J0000+0000"]

    info = source.info("basic,distance")

    assert "*** Basic info ***" in info
    assert "*** Distance info ***" in info
    assert "*** Position info ***" not in info
    assert "*** Derived parameters info ***" not in info


def test_source_str(psrcat_file):
    catalog = SourceCatalogPSRCAT(filename=psrcat_file)
    source = catalog["J0000+0000"]

    assert str(source) == source.info()
