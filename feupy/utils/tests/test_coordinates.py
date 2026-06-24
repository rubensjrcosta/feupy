# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from feupy.utils.coordinates import (
    convert_skycoord_to_dict,
    convert_dict_to_skycoord,
    convert_pos_config_to_skycoord,
    convert_table_to_skycoord,
)


# -------------------------
# SkyCoord -> dict
# -------------------------
def test_convert_skycoord_to_dict():
    coord = SkyCoord(10 * u.deg, 20 * u.deg, frame="icrs")

    result = convert_skycoord_to_dict(coord)

    assert result["frame"] == "icrs"
    assert result["lon"].deg == pytest.approx(10)
    assert result["lat"].deg == pytest.approx(20)


# -------------------------
# dict -> SkyCoord
# -------------------------
def test_convert_dict_to_skycoord():
    data = {
        "lon": 10 * u.deg,
        "lat": 20 * u.deg,
        "frame": "icrs",
    }

    coord = convert_dict_to_skycoord(data)

    assert coord.ra.deg == pytest.approx(10)
    assert coord.dec.deg == pytest.approx(20)


def test_convert_dict_missing_key():
    data = {"lon": 10 * u.deg, "frame": "icrs"}

    with pytest.raises(KeyError):
        convert_dict_to_skycoord(data)


# -------------------------
# pos_config -> SkyCoord
# -------------------------
class DummyPos:
    def __init__(self, lon=10 * u.deg, lat=20 * u.deg, frame="icrs"):
        self.lon = lon
        self.lat = lat
        self.frame = frame


def test_convert_pos_config_to_skycoord():
    pos = DummyPos()

    coord = convert_pos_config_to_skycoord(pos)

    assert coord.ra.deg == pytest.approx(10)
    assert coord.dec.deg == pytest.approx(20)


def test_convert_pos_config_missing_attr():
    class BadPos:
        def __init__(self):
            self.lon = 10

    with pytest.raises(AttributeError):
        convert_pos_config_to_skycoord(BadPos())


# -------------------------
# Table -> SkyCoord
# -------------------------
def test_convert_table_to_skycoord_ra_dec():
    table = Table()
    table["RA"] = [10, 20] * u.deg
    table["DEC"] = [30, 40] * u.deg

    coord = convert_table_to_skycoord(table)

    assert coord.ra.deg[0] == pytest.approx(10)
    assert coord.dec.deg[0] == pytest.approx(30)


def test_convert_table_to_skycoord_lowercase():
    table = Table()
    table["ra"] = [15] * u.deg
    table["dec"] = [25] * u.deg

    coord = convert_table_to_skycoord(table)

    assert coord.ra.deg == pytest.approx([15])
    assert coord.dec.deg == pytest.approx([25])


def test_convert_table_invalid():
    table = Table()
    table["X"] = [1, 2]
    table["Y"] = [3, 4]

    with pytest.raises(KeyError):
        convert_table_to_skycoord(table)