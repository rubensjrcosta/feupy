# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for coordinate conversions."""

from astropy.coordinates import SkyCoord
import astropy.units as u

import logging

log = logging.getLogger(__name__)

__all__ = [
    "convert_skycoord_to_dict",
    "convert_pos_config_to_skycoord",
    "convert_dict_to_skycoord",
    "convert_table_to_skycoord",
]


def convert_skycoord_to_dict(position: SkyCoord):
    """
    Convert a SkyCoord object to a dictionary.

    Parameters
    ----------
    position : SkyCoord
        Input sky coordinate.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'lon'
        - 'lat'
        - 'frame'
    """
    return {
        "lon": position.spherical.lon,
        "lat": position.spherical.lat,
        "frame": position.frame.name,
    }


def convert_pos_config_to_skycoord(pos_config):
    """
    Convert a configuration-like object to SkyCoord.

    The object must have attributes:
    - lon
    - lat
    - frame
    """
    required = ["lon", "lat", "frame"]

    for attr in required:
        if not hasattr(pos_config, attr):
            raise AttributeError(f"pos_config missing '{attr}'")

    return SkyCoord(pos_config.lon, pos_config.lat, frame=pos_config.frame)

def convert_dict_to_skycoord(pos_dict: dict):
    """
    Convert dictionary to SkyCoord.

    Supports:
    - lon/lat
    - ra/dec
    """

    # Detect keys automatically
    if {"lon", "lat"}.issubset(pos_dict):
        lon_key, lat_key = "lon", "lat"
    elif {"ra", "dec"}.issubset(pos_dict):
        lon_key, lat_key = "ra", "dec"
    else:
        raise KeyError(
            "Dictionary must contain ('lon','lat') or ('ra','dec')"
        )

    frame = pos_dict.get("frame", "icrs")

    lon = pos_dict[lon_key]
    lat = pos_dict[lat_key]

    # Handle missing units
    if not hasattr(lon, "unit"):
        lon = lon * u.deg
    if not hasattr(lat, "unit"):
        lat = lat * u.deg

    return SkyCoord(lon, lat, frame=frame)


def convert_table_to_skycoord(table):
    """
    Convert an astropy Table to SkyCoord.

    Supports multiple common column naming conventions.
    """
    keys = table.colnames

    if {"RAJ2000", "DEJ2000"}.issubset(keys):
        lon, lat, frame = "RAJ2000", "DEJ2000", "icrs"
    elif {"RAJ2000", "DECJ2000"}.issubset(keys):
        lon, lat, frame = "RAJ2000", "DECJ2000", "fk5"
    elif {"RA", "DEC"}.issubset(keys):
        lon, lat, frame = "RA", "DEC", "icrs"
    elif {"ra", "dec"}.issubset(keys):
        lon, lat, frame = "ra", "dec", "icrs"
    else:
        raise KeyError(
            "No valid coordinate columns found (RA/DEC, ra/dec, RAJ2000/DEJ2000)."
        )

    unit = table[lon].unit.to_string() if table[lon].unit else "deg"

    return SkyCoord(table[lon], table[lat], unit=unit, frame=frame)