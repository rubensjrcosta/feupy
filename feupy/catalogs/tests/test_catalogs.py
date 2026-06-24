# Licensed under a 3-clause BSD style license - see LICENSE.rst
from feupy.catalogs.utils import load_catalogs

def test_load_catalogs():
    catalogs = load_catalogs()
    assert catalogs is not None
    assert len(catalogs) > 0
