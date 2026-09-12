# Licensed under a 3-clause BSD style license - see LICENSE.rst
from feupy.irf import CTAOIRFManager


def test_irf_label():
    manager = CTAOIRFManager()

    opt = ("South", "AverageAz", "40deg", "50h")
    meta = manager.get_irf(opt)

    label = meta["label"]

    assert "South" in label
    assert "40deg" in label
    assert "50h" in label


def test_irf_name():
    manager = CTAOIRFManager()

    opt = ("North", "NorthAz", "20deg", "5h")
    meta = manager.get_irf(opt)

    name = meta["name"]

    assert "North" in name
    assert "20deg" in name
    assert "5h" in name


def test_irf_path():
    manager = CTAOIRFManager()

    opt = ("South", "AverageAz", "20deg", "0.5h")
    meta = manager.get_irf(opt)

    path = meta["file_path"]

    assert path.name.endswith(".fits.gz")
    assert "South" in str(path)


def test_irf_cache():
    manager = CTAOIRFManager()

    opt = ("South", "AverageAz", "20deg", "5h")

    meta1 = manager.get_irf(opt)
    meta2 = manager.get_irf(opt)

    assert meta1 is meta2


def test_observatory_selection():
    manager = CTAOIRFManager()

    opt_south = ("South", "AverageAz", "20deg", "5h")
    opt_north = ("North", "AverageAz", "20deg", "5h")

    meta_south = manager.get_irf(opt_south)
    meta_north = manager.get_irf(opt_north)

    assert meta_south["obs_location"] != meta_north["obs_location"]
