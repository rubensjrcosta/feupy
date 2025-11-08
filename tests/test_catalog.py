# Licensed under a 3-clause BSD style license - see LICENSE.rst
from feupy.catalog import load_catalog

def test_load_catalog():
    catalog = load_catalog("gamma_cat")
    assert catalog is not None
    assert len(catalog) > 0
