# Licensed under a 3-clause BSD style license - see LICENSE.rst
from feupy.analysis.irfs import Irfs

def test_irf_label():
    opt = ["South", "AverageAz", "40deg", "50h"]
    label = Irfs.get_irfs_label(opt)
    assert "South" in label
    assert "40deg" in label
    assert "50h" in label

