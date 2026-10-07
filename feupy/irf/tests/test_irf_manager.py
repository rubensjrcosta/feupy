# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest

from feupy.irf import CTAOIRFManager


# -------------------------------------------------------------------------
# Prod5 regression tests
# -------------------------------------------------------------------------


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


def test_prod5_default():
    manager = CTAOIRFManager()

    assert manager.production == "prod5"
    assert manager.version == "v0.1"


def test_prod5_observation_times():
    manager = CTAOIRFManager()

    assert set(manager.observation_times) == {
        "0.5h",
        "5h",
        "50h",
    }


def test_prod5_zeniths():
    manager = CTAOIRFManager()

    assert manager.zeniths == [
        "20deg",
        "40deg",
        "60deg",
    ]


def test_prod5_rejects_52deg():
    manager = CTAOIRFManager()

    opt = ("South", "AverageAz", "52deg", "0.5h")

    with pytest.raises(ValueError, match="Invalid zenith"):
        manager.get_irf(opt)


def test_prod5_rejects_100s():
    manager = CTAOIRFManager()

    opt = ("South", "AverageAz", "20deg", "100s")

    with pytest.raises(ValueError, match="Invalid livetime"):
        manager.get_irf(opt)


# -------------------------------------------------------------------------
# Prod6 tests
# -------------------------------------------------------------------------


def test_prod6_configuration():
    manager = CTAOIRFManager(production="prod6")

    assert manager.production == "prod6"
    assert manager.version == "v1.0"
    assert manager.condition == "dark"


def test_prod6_observation_times():
    manager = CTAOIRFManager(production="prod6")

    assert set(manager.observation_times) == {
        "100s",
        "0.5h",
        "5h",
        "50h",
    }


def test_prod6_zeniths():
    manager = CTAOIRFManager(production="prod6")

    assert manager.zeniths == [
        "20deg",
        "40deg",
        "52deg",
        "60deg",
    ]


def test_prod6_site_arrays():
    manager = CTAOIRFManager(production="prod6")

    assert manager.site_arrays == {
        "South": "2LSTs14MSTs37SSTs",
        "North": "4LSTs09MSTs",
    }


def test_prod6_path():
    manager = CTAOIRFManager(production="prod6")

    opt = ("South", "AverageAz", "20deg", "0.5h")
    meta = manager.get_irf(opt)

    path = meta["file_path"]

    assert path.name == (
        "Prod6-CTAO-South-20deg-AverageAz-"
        "2LSTs14MSTs37SSTs-dark-1800s-v1.0.fits.gz"
    )

    assert (
        path.parent.name
        == "CTAO-Performance-Prod6-CTAO-"
        "South-20deg-dark-v1.0.FITS"
    )


def test_prod6_load():
    manager = CTAOIRFManager(production="prod6")

    opt = ("South", "AverageAz", "20deg", "0.5h")
    meta = manager.get_irf(opt)

    assert set(meta["irf"]) == {
        "aeff",
        "psf",
        "edisp",
        "bkg",
    }

    assert meta["production"] == "prod6"
    assert meta["version"] == "v1.0"
    assert meta["condition"] == "dark"


def test_prod6_52deg_100s():
    manager = CTAOIRFManager(production="prod6")

    opt = ("South", "AverageAz", "52deg", "100s")
    meta = manager.get_irf(opt)

    assert set(meta["irf"]) == {
        "aeff",
        "psf",
        "edisp",
        "bkg",
    }

    assert "52deg" in str(meta["file_path"])
    assert "100s" in meta["file_path"].name


def test_prod6_halfmoon():
    manager = CTAOIRFManager(
        production="prod6",
        condition="halfmoon",
    )

    opt = ("South", "AverageAz", "20deg", "0.5h")
    meta = manager.get_irf(opt)

    assert meta["condition"] == "halfmoon"
    assert "halfmoon" in str(meta["file_path"])

    assert set(meta["irf"]) == {
        "aeff",
        "psf",
        "edisp",
        "bkg",
    }


# -------------------------------------------------------------------------
# Validation tests
# -------------------------------------------------------------------------


def test_invalid_production():
    with pytest.raises(
        ValueError,
        match="Invalid CTAO IRF production",
    ):
        CTAOIRFManager(production="invalid")


def test_invalid_prod6_condition():
    with pytest.raises(
        ValueError,
        match="Invalid Prod6 observing condition",
    ):
        CTAOIRFManager(
            production="prod6",
            condition="invalid",
        )


def test_prod6_rejects_prod5_subarray():
    manager = CTAOIRFManager(production="prod6")

    opt = (
        "South-MSTSubArray",
        "AverageAz",
        "20deg",
        "0.5h",
    )

    with pytest.raises(ValueError, match="Invalid array"):
        manager.get_irf(opt)


# -------------------------------------------------------------------------
# Available options
# -------------------------------------------------------------------------


def test_prod5_irf_options():
    manager = CTAOIRFManager()

    options = manager.get_irfs_options()

    # 6 arrays x 3 azimuths x 3 zeniths x 3 livetimes
    assert len(options) == 162

    assert (
        "South",
        "AverageAz",
        "20deg",
        "0.5h",
    ) in options


def test_prod6_irf_options():
    manager = CTAOIRFManager(production="prod6")

    options = manager.get_irfs_options()

    # 2 arrays x 3 azimuths x 4 zeniths x 4 livetimes
    assert len(options) == 96

    assert (
        "South",
        "AverageAz",
        "52deg",
        "100s",
    ) in options