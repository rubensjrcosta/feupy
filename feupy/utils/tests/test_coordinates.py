# Licensed under a 3-clause BSD style license - see LICENSE

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

from feupy.utils.coordinates import (
    convert_dict_to_skycoord,
    convert_pos_config_to_skycoord,
    convert_skycoord_to_dict,
    convert_table_to_skycoord,
)


class DummyPosition:
    def __init__(self, lon=10 * u.deg, lat=20 * u.deg, frame="icrs"):
        self.lon = lon
        self.lat = lat
        self.frame = frame


# -------------------------
# SkyCoord -> dict
# -------------------------
def test_convert_skycoord_to_dict():
    position = SkyCoord(10 * u.deg, 20 * u.deg, frame="icrs")

    result = convert_skycoord_to_dict(position)

    assert result["frame"] == "icrs"
    assert result["lon"].deg == pytest.approx(10)
    assert result["lat"].deg == pytest.approx(20)


# -------------------------
# dict -> SkyCoord
# -------------------------
def test_convert_dict_to_skycoord_lon_lat():
    position = {
        "lon": 10 * u.deg,
        "lat": 20 * u.deg,
        "frame": "icrs",
    }

    result = convert_dict_to_skycoord(position)

    assert result.ra.deg == pytest.approx(10)
    assert result.dec.deg == pytest.approx(20)


def test_convert_dict_to_skycoord_ra_dec():
    position = {
        "ra": 10 * u.deg,
        "dec": 20 * u.deg,
    }

    result = convert_dict_to_skycoord(position)

    assert result.frame.name == "icrs"
    assert result.ra.deg == pytest.approx(10)
    assert result.dec.deg == pytest.approx(20)


def test_convert_dict_to_skycoord_without_units():
    position = {
        "lon": 10,
        "lat": 20,
        "frame": "icrs",
    }

    result = convert_dict_to_skycoord(position)

    assert result.ra.deg == pytest.approx(10)
    assert result.dec.deg == pytest.approx(20)


def test_convert_dict_to_skycoord_missing_keys():
    position = {
        "lon": 10 * u.deg,
        "frame": "icrs",
    }

    with pytest.raises(KeyError, match="Dictionary must contain"):
        convert_dict_to_skycoord(position)


# -------------------------
# pos_config -> SkyCoord
# -------------------------
def test_convert_pos_config_to_skycoord():
    position = DummyPosition()

    result = convert_pos_config_to_skycoord(position)

    assert result.ra.deg == pytest.approx(10)
    assert result.dec.deg == pytest.approx(20)


def test_convert_pos_config_missing_attribute():
    class InvalidPosition:
        lon = 10 * u.deg

    with pytest.raises(AttributeError, match="pos_config missing"):
        convert_pos_config_to_skycoord(InvalidPosition())


# -------------------------
# Table -> SkyCoord
# -------------------------
@pytest.mark.parametrize(
    ("lon_name", "lat_name", "frame"),
    [
        ("RAJ2000", "DEJ2000", "icrs"),
        ("RAJ2000", "DECJ2000", "fk5"),
        ("RA", "DEC", "icrs"),
        ("ra", "dec", "icrs"),
    ],
)
def test_convert_table_to_skycoord(lon_name, lat_name, frame):
    table = Table()
    table[lon_name] = [10, 20] * u.deg
    table[lat_name] = [30, 40] * u.deg

    result = convert_table_to_skycoord(table)

    assert result.frame.name == frame
    assert result.spherical.lon.deg == pytest.approx([10, 20])
    assert result.spherical.lat.deg == pytest.approx([30, 40])


def test_convert_table_to_skycoord_without_units():
    table = Table()
    table["RA"] = [10, 20]
    table["DEC"] = [30, 40]

    result = convert_table_to_skycoord(table)

    assert result.ra.deg == pytest.approx([10, 20])
    assert result.dec.deg == pytest.approx([30, 40])


def test_convert_table_to_skycoord_invalid_columns():
    table = Table()
    table["X"] = [1, 2]
    table["Y"] = [3, 4]

    with pytest.raises(KeyError, match="No valid coordinate columns"):
        convert_table_to_skycoord(table)
